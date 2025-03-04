import os 
import json 
import yaml 
import csv 
import argparse
import matplotlib.pyplot as plt

# load the attributes of datasets
with open('datasets.json', 'r') as f:
    datasets_info = json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for eval.py")
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML config file')

    parser.add_argument('-e', '--eval_path', type=str, help='Path to the folder for model responses') # evaluation path if -c is not provided
    parser.add_argument('-p', '--plot', default=False, type=str, help='Show the plot') # show the plot  
    args = parser.parse_args()

    if args.config:
        with open('config.yaml', 'r') as file:
            configs = yaml.safe_load(file)

        dir = configs['eval_path']
    else:
        dir = args.eval_path

    conf_mat = False 
    total_acc = {} # dataset name: accuracy 
    for file in os.listdir(dir):
        dname = file[:file.find('.')]
        ds_info = datasets_info[dname]

        # get the available choices and name of answer column 
        choices = ds_info["choices"] 
        answer_col = ds_info["answer_col"]

        if dname == 'HaluEval': conf_mat, tp, fp, tn, fn = True, 0, 0, 0, 0
        else: correct = 0 

        if file.endswith("csv"): # for .csv file 
            with open(os.path.join(dir, file), 'r') as f:
                data = [{k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)]
                
        elif file.endswith("json"): # for .json file
            with open(os.path.join(dir, file), 'r') as f:
                data = json.load(f)
        
        else:
            raise ValueError("File type not supported")

        for row in data:
            if not row['processed_answer'].isalpha() or not (row['processed_answer'].lower() in choices or row['processed_answer'].upper() in choices): # unrelated responses 
                continue 

            if choices == ['A', 'B', 'C', 'D']: # multiple choice 
                ans = ord(row['processed_answer']) - 65
            else: # decision 
                ans = row['processed_answer'].lower()
            
            if dname == 'HaluEval':
                if ans == 'yes':
                    if ans == row[answer_col]:
                        tp += 1
                    else:
                        fp += 1
                elif ans == 'no':
                    if ans == row[answer_col]:
                        tn += 1
                    else:
                        fn += 1
                continue

            if ans == row[answer_col]:
                correct += 1
        
        if dname != 'HaluEval': 
            total_acc[dname] = correct/len(data)
    

    # Report the average accuracy for medical QA datasets 
    if len(total_acc) != 0:
        model_name = dir[dir.find('/')+1:dir.find('/', dir.find('/')+1)]
        print(f"Accuracy for medical QA datasets by {model_name}")
        for i, dname in enumerate(total_acc):
            print(f"{dname}: {total_acc[dname]}")

        print(f"Average: {sum(total_acc.values())/len(total_acc)}")

        if args.plot:
            # Plotting the bar chart
            d_name = list(total_acc.keys())
            acc = list(total_acc.values())
            colors = plt.cm.tab20.colors

            plt.figure(figsize=(10, 5))
            bars = plt.bar(d_name, acc, color=colors[:len(d_name)])
            plt.xlabel('Dataset')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy for Medical QA Datasets by {model_name}')
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
