import torch
import numpy as np
import random, os
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


from typing import List, Optional, Tuple, Type, Dict, TypeVar
from copy import deepcopy
import pathlib
import pickle
from tqdm import tqdm
import json, logging

T = TypeVar("T")
logger = logging.getLogger()
PROMPTS_ROOT = pathlib.Path('RRAG/prompts').resolve()

def get_qa_instruction(
    question: str, context: Dict, retrieval_aware: bool, RETRIEVAL_TOKEN, use_cot):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not context:
        raise ValueError(f"Provided `context` must be truthy, got: {context}")

    if retrieval_aware:
        prompt_filename = 'qa_similarity.prompt'
    elif use_cot:
        prompt_filename = 'qa_cot.prompt'
    else:
        prompt_filename = "qa.prompt"

    with open(prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    formatted_documents = []
    for document_index, document in enumerate(context):
        document_text = document['text']
        if retrieval_aware:
            document_prompt = f"[{document_index+1}]similarity: {RETRIEVAL_TOKEN}{document_text}"
        else:
            document_prompt = f"[{document_index+1}]{document_text}"
        formatted_documents.append(document_prompt)
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_instruction_dataset(dataset, max_prompt_length, tokenizer, retrieval_aware, use_cot, RETRIEVAL_TOKEN, sample_answer=True):
    instruction_dataset = []
    for input_example in tqdm(dataset):
        input_example = deepcopy(input_example)
        question = input_example["question"]
        context = input_example["ctxs"]
        if not context:
            raise ValueError(f"Did not find any documents for example: {input_example}")
        prompt = get_qa_instruction(
                question,
                context,
                retrieval_aware=retrieval_aware,
                RETRIEVAL_TOKEN=RETRIEVAL_TOKEN,
                use_cot=use_cot,
            )
        
        input_example['instruction'] = prompt

        answers = input_example['answers']
        for ans in answers:
            prompt_length = len(tokenizer(prompt + ans)["input_ids"])
            if max_prompt_length < prompt_length:
                print(
                            f"Skipping prompt ... with length {prompt_length}, which "
                            f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            data = deepcopy(input_example)
            data['output'] = ans
            instruction_dataset.append(data)
    return instruction_dataset

def get_embeds(dataset):
    for data in dataset:
        supporting_facts = data['supporting_facts']
        data['embeds'] = [[float(ctx['rerank_score']), float(ctx['nb_score']), float(ctx['precedent_score'])] for ctx in data['ctxs']]
        data['label'] = [1 if ctx['isgold'] else 0 for ctx in data['ctxs']]
    return dataset

def pre_dataset(dataset):
    remove_num = 0
    dataset_new = []
    for data in dataset:
        data = deepcopy(data)
        supporting_facts = []
        for ctx in data['ctxs']:
            if ctx['isgold']:
                supporting_facts.append(ctx['text'])
        if len(data['ctxs']) != 10:
            remove_num += 1
            continue
        data['supporting_facts'] = supporting_facts
        dataset_new.append(data)
    print('remove_num', remove_num)
    return dataset_new

def load_dataset_data(input_path):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)[:]
        f.close()
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)[:]
        f.close()
    train_data = pre_dataset(train_data[:])
    test_data = pre_dataset(test_data[:])
    print(f'prepare dataset, train size: {len(train_data)}, test size: {len(test_data)}')
    return train_data, test_data


def load_dataset(input_path, max_prompt_length, tokenizer, retrieval_aware, use_cot=False, RETRIEVAL_TOKEN='<R>'):
    train_data, test_data = load_dataset_data(input_path)
    instruction_dataset_train = get_instruction_dataset(train_data[:], max_prompt_length, tokenizer, retrieval_aware, use_cot, RETRIEVAL_TOKEN)
    instruction_dataset_test = get_instruction_dataset(test_data[:], max_prompt_length, tokenizer, retrieval_aware, use_cot, RETRIEVAL_TOKEN)

    instruction_dataset_train = get_embeds(instruction_dataset_train)
    instruction_dataset_test = get_embeds(instruction_dataset_test)
    return instruction_dataset_train, instruction_dataset_test

def get_dataset_ans(dataset):
    return [[data['answer']] for data in dataset]


