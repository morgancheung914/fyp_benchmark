import os 
import json 
import yaml 

with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

dir = configs['shortened']

total_acc = []
for file in os.listdir(dir):
    with open(os.path.join(dir, file), 'r') as f:
        data = json.load(f)

    correct = 0
    for i in data:
        if i["processed_answer"] in ["A", "B", "C", "D"]: # multiple choice answers 
            if file[:4] == 'MMLU':
                if ord(i["processed_answer"]) - 65 == i['answer']: # from MMLU
                    correct += 1

            else:
                if ord(i["processed_answer"]) - 65 == i['cop']: # from MedMCQA
                    
                    correct += 1

        elif i["processed_answer"] in ["Yes", "No", "Maybe"]: # MedMCQA
            if i["processed_answer"].lower() == i['final_decision']:
                correct += 1

    total_acc.append(correct/len(data))
    print(f"{file}: {total_acc[-1]}")
    

print(f"Average: {sum(total_acc)/len(total_acc)}")