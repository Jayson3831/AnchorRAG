import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List
from utils.logging_utils import logger
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI, AzureOpenAI
import os
import time

class LLMHandler:
    def __init__(self, llm, sbert):
        self.llm = llm
        # self.tokenizer, self.model = self._load_model()
        self.sbert = self._load_sbert(sbert)
    
    def _load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载LLM模型"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype="auto", 
                device_map="auto"
            )
            return tokenizer, model
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_text(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """使用LLM生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).split("\n", 1)[0]
    
    def qwen_generate_text(self, prompt: str, temperature: float=0.4):
        if not self.model:
            raise ValueError("LLM not initialized")
        messages = [
            {"role": "user", "content": prompt}
        ]

        tokenizer = self.tokenizer
        model = self.model
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            max_new_tokens=1024
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content

    def run_llm(self, prompt, args):
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
        message_prompt = {"role":"user","content":prompt}
        messages.append(message_prompt)

        if args.engine == "vllm":
            # 1. vllm部署LLM
            client = OpenAI(
                api_key=args.openai_api_keys,
                base_url=args.url,
            )
            completion = client.chat.completions.create(
                model="Qwen/Qwen3-8B",  # 按需更换为其它深度思考模型
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},      # vllm设置是否开启深度思考
                stream=False,
            )

        elif args.engine == "api":
            # 2. API部署LLM
            client = OpenAI(
                api_key=args.openai_api_keys,
                base_url=args.url,
            )

            try:
                completion = client.chat.completions.create(
                    model=self.llm,  # 按需更换为其它深度思考模型
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    extra_body={"enable_thinking": False},                                   # API设置是否开启深度思考
                    stream=False,
                )
            except Exception as e:
                logger.error(f"An error {e} occurred in the response process!")
                return "No response"

        elif args.engine == "azure_openai":
            # 3. Azure openai 部署 GPT4
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview",  # API 版本
                azure_endpoint="https://rage0612.openai.azure.com/",
            )

            try:
                completion = client.chat.completions.create(
                    model=self.llm,  # 用 Azure 模型部署名替换
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=False
                )
            except Exception as e:
                logger.error(f"An error {e} occurred in the response process!")
                return "No response"

        result = completion.choices[0].message.content
        if not result:
            return "No response"
        return result

    def _load_sbert(self, sbert) -> str:
        model = SentenceTransformer(sbert)
        return model
    
    def compute_similarity_batch(self, query: str, candidates: List[str]) -> List[float]:
        # 编码 query 和 candidates
        query_embedding = self.sbert.encode(query, convert_to_tensor=True, device='cuda', show_progress_bar=False)
        candidate_embeddings = self.sbert.encode(candidates, convert_to_tensor=True, device='cuda', show_progress_bar=False)

        # 计算余弦相似度，返回的是 tensor (1, N)
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        # 转换为 Python float 列表
        return cosine_scores.tolist()