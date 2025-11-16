import argparse
from tqdm import tqdm
from config.settings import (
    OUTPUT_PATH, JSON_PATH, IO_PROMPT, COT_PROMPT, SC_PROMPT, 
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
import time

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAGE")
    
    parser.add_argument("--dataset", type=str, default="webqsp_500", help="Select the dataset")
    parser.add_argument("--LLM", type=str, default='Qwen3-8B', help="LLM model name")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum output length for the LLM")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the large language model")
    parser.add_argument("--Sbert", type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Sentence-BERT model name or path")
    parser.add_argument("--openai_api_keys", type=str, default="", help="Your own OpenAI API keys.")
    parser.add_argument("--url", type=str, default="", help="Base URL.")
    parser.add_argument("--engine", type=str, default="vllm", help="Which platform you choose.")
    parser.add_argument("--width", type=int, default=3, help="Search width")
    parser.add_argument("--depth", type=int, default=3, help="Search depth")
    parser.add_argument("--num_retain_entity", type=int, default=10, help="Number of entities to retain")
    parser.add_argument("--keyword_num", type=int, default=5, help="Number of keywords for retrieval")
    parser.add_argument("--relation_num", type=int, default=5, help="Top-K relations per MID for similarity calculation")
    parser.add_argument("--prune_tools", type=str, default="sbert", choices=["llm", "sbert"], help="Pruning tool")
    parser.add_argument("--no-remove_unnecessary_rel", action="store_false", dest="remove_unnecessary_rel", help="Do not remove unnecessary relations")
    parser.add_argument("--method", type=str, default="rage", choices=['io', 'cot', 'sc', 'rage'], help="Method for experimental comparison")
    parser.add_argument("--agent_count", type=int, default=3, help="Number of agents")
    
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    
    fb_client = FreebaseClient()
    llm_handler = LLMHandler(args.LLM, args.Sbert)
    data_processor = DataProcessor(llm_handler)
    semantic_searcher = SemanticSearch()
    reasoning_engine = ReasoningEngine(fb_client, llm_handler, semantic_searcher)
    kg_system = KnowledgeGraphReasoningSystem(llm_handler, fb_client, semantic_searcher, args.agent_count)
    
    try:
        # load dataset
        logger.info(f"Loading {args.dataset} dataset...")
        datas, question_field = data_processor.load_dataset(args.dataset)
        logger.info(f"Loaded dataset: {args.dataset}, Samples: {len(datas)}")
        
        jsonl_file = OUTPUT_PATH.format(method=args.method, dataset=args.dataset, llm=args.LLM.split("/")[-1], suffix=args.prune_tools)
        json_file = JSON_PATH.format(method=args.method, dataset=args.dataset, llm=args.LLM.split("/")[-1], suffix=args.prune_tools)

        processed_questions = FileUtils.load_processed_questions(jsonl_file)
        
        logger.info("Retriving and generating...")
        start_time = time.time()
        llm_handler.reset_token_usage()
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
                    "candidate_entities": {},
                    "pre_heads": {},
                    "pre_relations": {},
                    "active_agents": list(range(1, args.agent_count + 1)),
                    "current_depth": 0,
                    "reasoning_chains": {},
                    "knowledge": {},
                    "whether_stop": False,
                    "final_answer": None,
                    "stop_reason": None,
                    "reasoning_log": [],
                    "output_file": jsonl_file
                }

                kg_system.workflow.invoke(initial_state)

        logger.info("In the assessment results...")
        FileUtils.jsonl2json(jsonl_file, json_file)
        eval_em(args.dataset, json_file, args.LLM, args.method)

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise

    finally:
        # 5. 结束计时并打印统计信息（无论成功或失败）
        end_time = time.time()
        elapsed_time = end_time - start_time
        token_usage = llm_handler.get_token_usage()
        
        logger.info("\n" + "="*50)
        logger.info("Execution Statistics:")
        logger.info(f"  - Total Execution Time: {elapsed_time:.2f} seconds")
        if token_usage['total_tokens'] > 0:
            logger.info(f"  - Total LLM Tokens Used: {token_usage['total_tokens']}")
            logger.info(f"    - Prompt Tokens: {token_usage['prompt_tokens']}")
            logger.info(f"    - Completion Tokens: {token_usage['completion_tokens']}")
        else:
            logger.info("  - No LLM tokens were used in this run.")
        logger.info("="*50 + "\n")

if __name__ == '__main__':
    main()