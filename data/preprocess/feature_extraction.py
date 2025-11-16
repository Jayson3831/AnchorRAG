import torch
import numpy as np
import random, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.set_device(0)
print(torch.cuda.is_available())
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(42)

import pickle
from tqdm import tqdm
import json
import argparse
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import RerankingEvaluator
from tqdm import tqdm, trange


os.chdir(sys.path[0])

### webqsp ### ### cwq ###
def load_data_json(input_path, dataset_seed=42):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        f.close()
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        f.close()
    print(f'prepare dataset, train size: {len(train_data)}, test size: {len(test_data)}')
    return train_data, test_data

def get_dual_dev(dataset):
    dev_data = []
    for data in dataset:
        query = data['question']
        ctxs = data['ctxs']
        pos = []
        neg = []
        for ctx in ctxs:
            text = ctx.get('text')
            if ctx['isgold']:
                pos.append(text)
            else:
                neg.append(text)
        dev_data.append({'query': query, 'positive': pos, 'negative': neg})
    return dev_data

def get_dual_sim(dataset, retrival_model):
    dataset_new = []
    for data in tqdm(dataset):
        data = deepcopy(data)
        query = data['question']
        ctxs = data['ctxs']
        ctxs_text = []
        for ctx in ctxs:
            text = ctx.get('text')
            ctxs_text.append(text)
        embs = retrival_model.encode([query] + ctxs_text, convert_to_tensor=True)
        q_emb = embs[0]
        c_emb = embs[1:]
        scores, rank_list, precendent_scores = get_precedent_sim(q_emb, c_emb)
        nb_scores = get_nb_sim(c_emb, rank_list)
        ctxs = []
        for i in range(len(rank_list)):
            j = rank_list[i]
            ctx = deepcopy(data['ctxs'][j])
            rerank_score = float(scores[j])
            rerank_nb_score = float(nb_scores[i])
            rerank_precedent_score = float(precendent_scores[i])
            ctx['rerank_score'] = rerank_score
            ctx['nb_score'] = rerank_nb_score
            ctx['precedent_score'] = rerank_precedent_score
            ctxs.append(ctx)
        data['ctxs'] = ctxs
        dataset_new.append(data)
    return dataset_new

def get_precedent_sim(q_emb, c_emb):
    cosine_score = cos_sim(q_emb, c_emb)[0].cpu()
    rank_list = np.array(cosine_score).argsort().tolist()[::-1]
    precedent_sim = [1]
    for i in range(1, len(rank_list)):
        precedent_score = np.array([cosine_score[idx] for idx in rank_list[:i]])
        precedent_weight = np.exp(precedent_score)/np.exp(precedent_score).sum()
        w = torch.Tensor(precedent_weight, device='cpu').reshape(1, precedent_score.shape[0])
        precedent_embs = [c_emb[idx].reshape(1, c_emb.shape[1]).cpu() for idx in rank_list[:i]]
        precedent_emb = torch.mm(w, torch.cat(precedent_embs))
        cur_emb = c_emb[rank_list[i]].cpu()
        score = cos_sim(cur_emb, precedent_emb)[0][0]
        precedent_sim.append(score)
    return np.array(cosine_score), rank_list, np.array(precedent_sim)

def get_nb_sim(c_emb, rank_list):
    rank_emb = [c_emb[i] for i in rank_list]
    nb_sim = [cos_sim(rank_emb[0], rank_emb[1]).cpu()]
    for i in range(1, len(rank_list)-1):
        nb_sim.append((cos_sim(rank_emb[i], rank_emb[i-1]).cpu() + cos_sim(rank_emb[i], rank_emb[i+1]).cpu()) / 2)
    nb_sim.append(cos_sim(rank_emb[-2], rank_emb[-1]).cpu())
    nb_sim = [s.squeeze() for s in nb_sim]
    return nb_sim

##### Feature Extration and Save
def main(dataset_name, train_data_path, test_data_path, model_name, save_path, save_train_path, save_test_path, output_path):
    ##### Load Retriever
    model = SentenceTransformer(model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ### webqsp ### ### cwq ### ### grailqa ###
    if dataset_name == 'webqsp' or dataset_name == 'cwq' or dataset_name == 'grailqa':
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
        train_data, test_data = load_data_json(input_path)

        # #### Evaluate on dev set
        # test_samples = get_dual_dev(test_data)
        # dev_evaluator = RerankingEvaluator(test_samples, batch_size=32, show_progress_bar=True)
        # r = dev_evaluator(model, output_path) 

        dataset = get_dual_sim(train_data, model)
        dataset_test = get_dual_sim(test_data, model)
        with open(save_train_path, 'wb') as fin:
            pickle.dump(dataset, fin)
            fin.close()
        with open(save_test_path, 'wb') as fin:
            pickle.dump(dataset_test, fin)
            fin.close()

    else:
        raise ValueError(dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval feature extraction")
    parser.add_argument('--dataset_name', type=str, default='webqsp', choices=['webqsp', 'cwq', 'grailqa', 'hotpotqa'], help='Name of the dataset to process')    
    parser.add_argument('--train_data_path', type=str, required=False, default='../webqsp/webqsp_with_triples.train.json', help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--test_data_path', type=str, required=False, default='../webqsp/webqsp_with_triples.test.json', help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceTransformer model name or path for retriever')
    parser.add_argument('--save_path', type=str, required=False, default='../webqsp/webqsp.pkl', help='Path to save train dataset')
    parser.add_argument('--save_train_path', type=str, required=False, default='../webqsp/webqsp.train.pkl', help='Path to save train dataset')
    parser.add_argument('--save_test_path', type=str, required=False, default='../webqsp/webqsp.test.pkl', help='Path to save test dataset')
    parser.add_argument('--output_path', type=str, required=False, default='../temp_result', help='Path to save the evluation results')
    args = parser.parse_args()

    main(args.dataset_name, args.train_data_path, args.test_data_path, args.model_name, args.save_path, args.save_train_path, args.save_test_path, args.output_path)