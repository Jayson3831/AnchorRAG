import json
import re


def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'cwq_1000':
        with open('../data/cwq/cwq_1000.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'cwq_dt_0.2':
        with open('../data/cwq/cwq_with_dt_0.2.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'webqsp_500':
        with open('../data/webqsp/webqsp_500.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'webqsp_dt_0.2':
        with open('../data/webqsp/webqsp_dt_0.2.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'noisy_webqsp':
        with open('../data/noisy_WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'Noisy_Question'
    elif dataset_name == 'noisy_grailqa':
        with open('../data/noisy_grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'Noisy_Question'
    elif dataset_name == 'noisy_cwq':
        with open('../data/noisy_cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'Noisy_Question'
    elif  dataset_name == 'noisy_webq':
        with open('../data/noisy_WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'Noisy_Question'
    elif dataset_name == 'webqsp_sampled':
        with open('../data/WebQSP_sampled_600.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'grailqa_100':
        with open('../data/grailqa/grailqa_100.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webq':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    elif dataset_name == 'ow_webqsp':
        with open('../data/OW_webqsp.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'ow_grailqa':
        with open('../data/OW_grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    with open(output_file, encoding='utf-8') as f:
        output_datas = json.load(f)
    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas):
    answer_list= []
    origin_data = [j for j in ground_truth_datas if j[question_string] == data['question']][0]
    if dataset_name == 'cwq' or dataset_name == 'noisy_cwq' or dataset_name == 'ow_grailqa' or dataset_name == 'ow_webqsp' or 'cwq' in dataset_name:
        answer = origin_data["answer"]
        answer_list.append(answer)

    elif dataset_name == 'webqsp' or dataset_name == 'noisy_webqsp' or 'webqsp' in dataset_name:
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa' or dataset_name == 'noisy_grailqa' or 'grailqa' in dataset_name:
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
        
    elif dataset_name == 'webq' or dataset_name == 'noisy_webq':
        answer_list = origin_data["answers"]

    elif dataset_name == 'trex' or dataset_name == 'zeroshotre':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'creak':
        answer = origin_data['label']
        answer_list.append(answer)

    return list(set(answer_list))
    
def check_string(string):
    return "{" in string

# def clean_results(string):
#     if "{" in string:
#         start = string.find("{") + 1
#         end = string.find("}")
#         content = string[start:end]
#         return content
#     else:
#         return "NULL"
    
def clean_results(string):
    matches = re.findall(r"\{(.*?)\}", string)
    if not matches:
        return string  # 没有匹配项
    if matches[0] == "Yes":
        if len(matches) > 1:
            return matches[1]
        return string
    return matches[0]  # 不是 Yes，就取第一个

def check_refuse(string):
    refuse_words = ["however", "sorry"]
    return any(word in string.lower() for word in refuse_words)


def exact_match(response, answer):
    if isinstance(response, list):
        cleaned_list = [item.strip().lower() for item in response]
        clean_result = ",".join(cleaned_list)
    elif isinstance(response, str):
        clean_result = response.strip().replace(" ","").lower()
    clean_answer = answer.strip().replace(" ","").lower()
    if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
        return True
    return False

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

def save_result2json(dataset_name, Hit, Accuracy, model, method):
    results_data = {
        'dataset': dataset_name,
        'model': model,
        'method': method,
        'Hit@1': Hit,
        'Accuracy': Accuracy
    }
    with open('../results/{}_{}_{}_acc.json'.format(method, dataset_name, model), 'a', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

def check_wrong_result(method, dataset, model, data):
    with open('../wrong_sample/{}_{}_{}_error.json'.format(method, dataset, model), 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_content(s):
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'