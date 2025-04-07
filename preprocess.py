from datasets import load_dataset
import json

with open('datasets.json', 'r') as f:
    datasets_info = json.load(f)

def process_data(dataset, cache_dir, number_shot=0, CoT=0):
    dname_to_ds = {}
    for dname in dataset:
        if dname not in datasets_info:
            raise ValueError(f"Dataset {dname} is not supported.")
        
        ds_info = datasets_info[dname]
        question = ds_info['question']
        choice = ds_info['choice']
        bm_split = ds_info['benchmark_split'] # obtain the benchmarking split
        subset = ds_info['subset']

        ds = load_dataset(ds_info['path'], subset, cache_dir=cache_dir)

        if dname == 'PubMedQA':
            ds = ds['train'].train_test_split(test_size=0.5, shuffle=False) # split into train, test sets

        # user prompt processing 
        if choice is None:
            ds[bm_split] = ds[bm_split].add_column(name="user_content", column=[
                f"Question: {i[question]}" for i in ds[bm_split]
            ])
        elif len(choice) == 1: 
            ds[bm_split] = ds[bm_split].add_column(name="user_content", column=[
                f"Question: {i[question]}, Choices: A. {i[choice[0]][0]}, B. {i[choice[0]][1]}, C. {i[choice[0]][2]}, D. {i[choice[0]][3]}" for i in ds[bm_split]
            ])
        else: 
            ds[bm_split] = ds[bm_split].add_column(name="user_content", column=[
                f"Question: {i[question]}, Choices: A. {i[choice[0]]}, B. {i[choice[1]]}, C. {i[choice[2]]}, D. {i[choice[3]]}" for i in ds[bm_split]
            ])

        # system prompt processing 
        sys_prompt = ds_info['sys_prompt']

        if CoT: # Chain of thought prompt processing
            sys_prompt += " Here I will give you few examples: \n"

            CoT_path = ds_info["CoT_path"]
            if CoT_path is None:
                raise ValueError(f"Chain of Thoughts for {dname} is not supported.")
            with open(ds_info['CoT_path'], "r") as f:
                CoT_prompt = f.read()
        
            sys_prompt += CoT_prompt
            sys_prompt += "Now please answer the user's question by thinking step by step."

        elif number_shot: # few-shot prompt processing
            sys_prompt += " Here I will give you few examples: \n"
            
            fs_split = ds_info["fs_split"]
            fs_ds = ds[fs_split]

            if fs_split is None:
                raise ValueError(f"Few-shot prompting for {dname} is not supported.")
            for i in range(number_shot):
                sys_prompt += f"Question {i+1}: {fs_ds[i]['user_content']}\nAnswer {i+1}: {chr(fs_ds[i]['cop'] + 65)}\n"
            
            sys_prompt += "Now please answer the user's question."
        
        ds[bm_split] = ds[bm_split].add_column(name="sys_content", column=[sys_prompt] * len(ds[bm_split]))

        if dname == 'MedMCQA': dname_to_ds[dname] = ds[bm_split].select(range(1000))  
        else: dname_to_ds[dname] = ds[bm_split]

    return dname_to_ds
