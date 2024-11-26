import os 
import json 
import yaml 
import csv 

with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

dir = configs['shortened']

total_acc = []
for file in os.listdir(dir):
    if file[:8] == 'HaluEval': tp, fp, tn, fn = 0, 0, 0, 0
    else: correct = 0 

    if file.endswith("json"): # for .json file 
        with open(os.path.join(dir, file), 'r') as f:
            data = json.load(f)

    elif file.endswith("csv"): # for .csv file 
        with open(os.path.join(dir, file), 'r') as f:
            data = [{k: v for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)]

    for i in data:
        if i["processed_answer"] in ["A", "B", "C", "D"]: # valid multiple choice answers 
            if file[:4] == 'MMLU': # from MMLU datasets
                if ord(i["processed_answer"]) - 65 == int(i['answer']):
                    correct += 1

            else: # from MedMCQA
                if ord(i["processed_answer"]) - 65 == int(i['cop']): 
                    correct += 1

        elif i["processed_answer"].lower() in ["yes", "no", "maybe"]: 
            if file[:8] == 'PubMedQA': # from PubMedQA
                if i["processed_answer"].lower() == i['final_decision']:
                    correct += 1
        
            elif file[:8] == 'HaluEval': # from HaluEval
                if i["hallucination"] == 'yes':
                    if i["processed_answer"] == i["hallucination"]:
                        tp += 1
                    else:
                        fp += 1
                elif i["hallucination"] == 'no':
                    if i["processed_answer"] == i["hallucination"]:
                        tn += 1
                    else:
                        fn += 1

    if file[:8] != 'HaluEval': 
        total_acc.append(correct/len(data))
        print(f"{file}: {total_acc[-1]}")

# Report the average accuracy for medical QA datasets 
print(f"Average accuracy for medical QA datasets: {sum(total_acc)/len(total_acc)}")

# # Report the confusion matrix from the HaluEval dataset
# print(f"""Confusion Matrix for HaluEval:
#     yes    no 
# yes  {tp:<3} | {fn:<3}
# no   {fp:<3} | {tn:<3}""")
# print(f"Accuracy for HaluEval: {(tp+tn)/(tp+fn+fp+tn)}")
