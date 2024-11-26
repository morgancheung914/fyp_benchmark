import os 
import json 
import yaml 
import csv 
import random

with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

dir = configs['shortened']
self_con = configs['generation']['k_self_consistency']
total_acc = []

def majority_voting(i):
    """
    Takes a row of the dataset, unpacks the json string and do majority voting, and set the result as processed_answer
    Args:
        i: a row of the dataset
    
    """
    k_ans = json.loads(i['processed_answer'])   
    voting = {}
    for i in range(len(k_ans)):
        if k_ans[i] not in voting.keys():
            voting[k_ans[i]] = 1 
        else:
            voting[k_ans[i]] += 1

    max_value = max(voting.values())
    keys_with_max_value = [key for key, value in voting.items() if value == max_value]

    f_ans = random.choice(keys_with_max_value)

    
    print(f"k_paths: {k_ans}, chosen: {f_ans}")
    return f_ans
for file in os.listdir(dir):
    if file[:8] == 'HaluEval': conf_mat, tp, fp, tn, fn = True, 0, 0, 0, 0
    else: conf_mat, correct = False, 0 

    if file.endswith("json"): # for .json file 
        with open(os.path.join(dir, file), 'r') as f:
            data = json.load(f)

    elif file.endswith("csv"): # for .csv file 
        with open(os.path.join(dir, file), 'r') as f:
            data = [{k: v for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)]

    for i in data:
        # handle majority voting in self-consistency
        if self_con != False:
            f_ans = majority_voting(i)
                
            polished_answer = f_ans
        else:
            polished_answer = i['processed_answer']

        if polished_answer in ["A", "B", "C", "D"]: # valid multiple choice answers 
            if file[:4] == 'MMLU': # from MMLU datasets
                if ord(polished_answer) - 65 == int(i['answer']):
                    correct += 1

            else: # from MedMCQA
                if ord(polished_answer) - 65 == int(i['cop']): 
                    correct += 1

        elif polished_answer.lower() in ["yes", "no", "maybe"]: 
            if file[:8] == 'PubMedQA': # from PubMedQA
                if polished_answer.lower() == i['final_decision']:
                    correct += 1
        
            elif file[:8] == 'HaluEval': # from HaluEval
                if i["hallucination"] == 'yes':
                    if polished_answer == i["hallucination"]:
                        tp += 1
                    else:
                        fp += 1
                elif i["hallucination"] == 'no':
                    if polished_answer == i["hallucination"]:
                        tn += 1
                    else:
                        fn += 1

    if file[:8] != 'HaluEval': 
        total_acc.append(correct/len(data))
        print(f"{file}: {total_acc[-1]}")

# Report the average accuracy for medical QA datasets 
print(f"Average accuracy for medical QA datasets: {sum(total_acc)/len(total_acc)}")

# Report the confusion matrix from the HaluEval dataset
if conf_mat:
    print(f"""Confusion Matrix for HaluEval:
        yes    no 
    yes  {tp:<3} | {fn:<3}
    no   {fp:<3} | {tn:<3}""")
    print(f"Accuracy for HaluEval: {(tp+tn)/(tp+fn+fp+tn)}")
