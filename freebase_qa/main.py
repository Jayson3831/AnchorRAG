import argparse
from tqdm import tqdm
from config.settings import (
    OUTPUT_PATH, JSON_PATH, IO_PROMPT, COT_PROMPT, SC_PROMPT
)
from core.freebase_client import FreebaseClient
from core.llm_handler import LLMHandler
from core.data_processor import DataProcessor
from core.reasoning_engine import ReasoningEngine
from core.semantic_search import SemanticSearch
from core.lg_multi_agent import KnowledgeGraphReasoningSystem
from eval import eval_em
from utils.file_utils import FileUtils
from utils.logging_utils import setup_logging, logger
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.chdir(sys.path[0])  #使用文件所在目录
# print(f"Available GPUs: {torch.cuda.device_count()}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAGE")
    
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="webqsp",
                       help="选择数据集")
    
    # 模型参数
    parser.add_argument("--LLM", type=str, default='qwen-plus',
                       help="LLM模型名称")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="LLM输出最大长度")
    parser.add_argument("--temperature", type=float, 
                       default=0.7, help="大模型温度系数")
    parser.add_argument("--Sbert", type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help="LLM模型名称")
    parser.add_argument("--openai_api_keys", type=str,
                        default=os.getenv("DASHSCOPE_API_KEY"), help="your own openai api keys.")
    parser.add_argument("--url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="base url.")
    parser.add_argument("--engine", type=str,
                        default="api", help="which platform you choose.")

    # 搜索参数
    parser.add_argument("--width", type=int, default=3,
                       help="宽度")
    parser.add_argument("--depth", type=int, default=3,
                       help="搜索深度")
    parser.add_argument("--num_retain_entity", type=int,
                       default=10,
                       help="保留的实体数量")
    parser.add_argument("--keyword_num", type=int, default=5,
                       help="关键词检索数量")
    parser.add_argument("--relation_num", type=int, default=5,
                       help="每个mid的Top-K关系相似度来求平均")

    # 剪枝选项
    parser.add_argument("--prune_tools", type=str, default="llm",
                       choices=["llm", "sentencebert"],
                       help="剪枝工具")
    parser.add_argument("--no-remove_unnecessary_rel", action="store_false",
                       dest="remove_unnecessary_rel",
                       help="不移除不必要的关系")
    
    # 对比方法
    parser.add_argument("--method", type=str, default="rage",
                        choices=['io', 'cot', 'sc', 'rage'], help="实验对比方法")
    
    # 智能体参数
    parser.add_argument("--agent_count", type=int, default=3, help="实验对比方法")
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    
    # 初始化组件
    fb_client = FreebaseClient()
    llm_handler = LLMHandler(args.LLM, args.Sbert)
    data_processor = DataProcessor(llm_handler)
    semantic_searcher = SemanticSearch()
    reasoning_engine = ReasoningEngine(fb_client, llm_handler, semantic_searcher)
    kg_system = KnowledgeGraphReasoningSystem(llm_handler, fb_client, semantic_searcher, args.agent_count)
    
    try:
        # 加载数据集
        logger.info(f"Loading {args.dataset} dataset...")
        datas, question_field = data_processor.load_dataset(args.dataset)
        logger.info(f"Loaded dataset: {args.dataset}, Samples: {len(datas)}")
        
        # 准备输出文件
        jsonl_file = OUTPUT_PATH.format(method=args.method, dataset=args.dataset, suffix=args.LLM.split("/")[-1])
        json_file = JSON_PATH.format(method=args.method, dataset=args.dataset, suffix=args.LLM.split("/")[-1])
        processed_questions = FileUtils.load_processed_questions(jsonl_file)
        
        # 处理问题
        logger.info("Retriving and generating...")
        for data in tqdm(datas, desc=f"{args.method}..."):
            question = data[question_field]
            
            if question in processed_questions:
                continue

            if args.method == "io":
                prompt = IO_PROMPT + "\n\nQ: " + question + "\nA: "
                results = reasoning_engine.llm.run_llm(prompt, args)
                reasoning_engine.save_results(question, results, [], jsonl_file)

            elif args.method == "cot":
                prompt = COT_PROMPT + "\n\nQ: " + question + "\nA: "
                results = reasoning_engine.llm.run_llm(prompt, args)
                reasoning_engine.save_results(question, results, [], jsonl_file)
                
            elif args.method == "sc":
                cot_prompt = COT_PROMPT + "\n\nQ: " + question + "\nA: "
                inference_num = 5
                results = []
                for i in range(inference_num):
                    try:
                        response = reasoning_engine.llm.run_llm(cot_prompt, args)
                        results.append(response)
                    except Exception as e:
                        logger.error(f"Error during generation {i + 1}: {e}")
                        results.append("Error")
                reasoning_paths = "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(results))
                sc_prompt = SC_PROMPT.format(question, reasoning_paths)
                sc_result = reasoning_engine.llm.run_llm(sc_prompt, args)
                reasoning_engine.save_results(question, sc_result, [], jsonl_file)

            elif args.method == "rage":
                initial_state = {
                    "args": args,
                    "question": question,
                    "candidate_entities": {},           # 由 filter 智能体填充 agent_id -> {mid: name}
                    "pre_heads": {},                    # 由 filter 智能体初始化 agent_id -> [-1, ...]
                    "pre_relations": {},               # 由 filter 智能体初始化 agent_id -> []
                    "active_agents": list(range(1, args.agent_count + 1)),
                    "current_depth": 0,                # 设置为 0，filter 执行后自动设为 1
                    "reasoning_chains": {},            # agent_id -> [[triplet_chain]]
                    "knowledge": {},                   # agent_id -> [summary]
                    "whether_stop": False,
                    "final_answer": None,
                    "stop_reason": None,
                    "reasoning_log": [],
                    "output_file": jsonl_file
                }

                # 运行流程
                result = kg_system.workflow.invoke(initial_state)
                # print("Final Answer:", result["final_answer"])
        
        logger.info("In the assessment results...")
        FileUtils.jsonl2json(jsonl_file, json_file)
        eval_em(args.dataset, json_file, args.LLM, args.method)

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()