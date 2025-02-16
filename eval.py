import os 
import json 
import yaml 
import csv 
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Argument parser for eval.py")
parser.add_argument('-c', '--config', type=str, default=None, help='Path to the YAML config file')

# Arguments for response path if -c is not provided
parser.add_argument('-e', '--eval_path', type=str, help='Path to the folder for model responses')

args = parser.parse_args()
if args.config:
    with open('config.yaml', 'r') as file:
            configs = yaml.safe_load(file)

    dir = configs['eval']
else:
    dir = args.eval_path

conf_mat = False 
total_acc = {}
for file in os.listdir(dir):
    if file[:8] == 'HaluEval': conf_mat, tp, fp, tn, fn = True, 0, 0, 0, 0
    else: correct = 0 

    if file.endswith("csv"): # for .csv file 
        with open(os.path.join(dir, file), 'r') as f:
            data = [{k: v for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)]
    else: # for .json file
        with open(os.path.join(dir, file), 'r') as f:
            data = json.load(f)

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
                if i["processed_answer"] == 'yes':
                    if i["processed_answer"] == i["hallucination"]:
                        tp += 1
                    else:
                        fp += 1
                elif i["processed_answer"] == 'no':
                    if i["processed_answer"] == i["hallucination"]:
                        tn += 1
                    else:
                        fn += 1

    if file[:8] != 'HaluEval': 
        total_acc[file[:file.find('.')]] = correct/len(data)
  

# Report the average accuracy for medical QA datasets 
if total_acc != []:
    model = dir[dir.find('/')+1:dir.find('/',dir.find('/')+1)]
    print(f"Average accuracy for medical QA datasets: {sum(total_acc.values())/len(total_acc)}")

    # Plotting the bar chart
    d_name = list(total_acc.keys())
    acc = list(total_acc.values())
    colors = plt.cm.tab20.colors

    plt.figure(figsize=(10, 5))
    bars = plt.bar(d_name, acc, color=colors[:len(d_name)])
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy for Medical QA Datasets by {model}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.legend(bars, d_name, title="Datasets")
    plt.show()

# Report the confusion matrix from the HaluEval dataset
if conf_mat:
    print(f"""Confusion Matrix for HaluEval:
       yes   no 
yes  {tp:<3} | {fn:<3}
no   {fp:<3} | {tn:<3}""")
    print(f"Accuracy for HaluEval: {(tp+tn)/(tp+fn+fp+tn)}")
