from utils.eval_utils import *
from utils.file_utils import FileUtils
import os, sys
os.chdir(sys.path[0])  #使用文件所在目录

def eval_em(dataset, output_file, model_name, method, constraints_refuse=True):
    model_stem = model_name.split("/")[-1]
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(dataset, output_file)

    acc_list = []
    hit_list = []
    for data in output_datas:
        answers = align(dataset, question_string, data, ground_truth_datas)
        results = data['results']
        if check_string(results):
            response = clean_results(results)
            hit = eval_hit(response, answers)
            acc = eval_acc(response, answers)
            hit_list.append(hit)
            acc_list.append(acc)
        else:
            response = results
            if constraints_refuse and check_string(response):
                continue
            hit = eval_hit(response, answers)
            acc = eval_acc(response, answers)
            hit_list.append(hit)
            acc_list.append(acc)

    Hit = sum(hit_list) * 100 / len(hit_list)
    Accuracy = sum(acc_list) * 100 / len(acc_list)
    print("Hit: " + str(Hit))
    print("Accuracy: " + str(Accuracy))

    save_result2json(dataset, Hit, Accuracy, model_stem, method)

def eval_hit(prediction, answers):
    for answer in answers:
        if exact_match(prediction, answer):
            return 1
    return 0

def eval_acc(prediction, answers):
    matched = 0
    if len(answers) == 0:
        return 0
    for answer in answers:
        if exact_match(prediction, answer):
            matched += 1
    return matched / len(answers)

if __name__ == '__main__':
    dataset = 'webqsp'
    model = 'gpt-4o-mini'
    method = 'PoG'
    json_file = f"../outputs/{method}_{dataset}_{model}.json"
    jsonl_file = f"../outputs/{method}_{dataset}_{model}.jsonl"
    
    FileUtils.jsonl2json(jsonl_file, json_file)
    eval_em(dataset, json_file, model, method)
    