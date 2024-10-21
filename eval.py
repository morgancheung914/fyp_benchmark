import os 
import json 
import yaml 

with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

dir = configs['shortened']

for file in os.listdir(dir):
    with open(os.path.join(dir, file), 'r') as f:
        data = json.load(f)

    correct = 0
    for i in data:
        if file[:4] == 'MMLU':
            if i["processed_answer"] not in ["A", "B", "C", "D", "Yes", "No", "Maybe"]: continue # ignore appropriate reponse 
            
            elif ord(i["processed_answer"]) - 65 == i['answer']:
                correct += 1
        elif file == 'PubMedQA':
            if i["processed_answer"] not in ["A", "B", "C", "D", "Yes", "No", "Maybe"]: continue # ignore appropriate reponse 
            
            elif i["processed_answer"].lower() == i['final_decision']:
                correct += 1
        else:
            if i["processed_answer"] not in ["A", "B", "C", "D", "Yes", "No", "Maybe"]: continue # ignore appropriate reponse 
            
            elif ord(i["processed_answer"]) - 65 == i['cop']:
                correct += 1
    print(f"{file}: accuracy is {correct/len(data)}")