from datasets import load_dataset
import argparse 
import re 

def MMLU_formatter(ds, number_shot):
        # adding user content and system content 
        ds['validation'] = ds['validation'].add_column(name="user_content", column=[
            f"Question: {i['question']}, Choices: A: {i['choices'][0]}, B: {i['choices'][1]}, C: {i['choices'][2]}, D: {i['choices'][3]}" for i in ds['validation']])
        ds['test'] = ds['test'].add_column(name="user_content", column=[
            f"Question: {i['question']}, Choices: A: {i['choices'][0]}, B: {i['choices'][1]}, C: {i['choices'][2]}, D: {i['choices'][3]}" for i in ds['test']])
        
        sys_prompt = "Please read the question and pick the most suitable choice from A to D, simply answer A, B, C or D."
        if number_shot: # few-shot preprocessing 
            sys_prompt += " Here I will give you few examples: \n"
            fs_ds = ds["validation"]

            for i in range(number_shot):
                
                sys_prompt += f"Question {i+1}: {fs_ds[i]['user_content']}\nAnswer {i+1}: {chr(fs_ds[i]['answer'] + 65)}\n"
            
            sys_prompt += "Now please answer the user's question."
        print(sys_prompt)
        ds['test'] = ds['test'].add_column(name="sys_content", column=[sys_prompt] * len(ds['test']))
    
         
        return ds 


def process_data(dataset, cache_dir, number_shot, CoT): # takes in dataset and cache_dir configs 
    PubMedQA_ds, MedMCQA_ds, anatomy_ds, biology_ds, medicine_ds, clinical_ds, halu_ds = None, None, None, None, None, None, None
    
    # {'pub_id', 'question', 'context', 'long_answer', 'final_decision'}
    if 'PubMedQA' in dataset:
        PubMedQA_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train', cache_dir=cache_dir)
        PubMedQA_ds = PubMedQA_ds.train_test_split(test_size=0.5) # train, test 
        
        # adding user content and system content 
        PubMedQA_ds['train'] = PubMedQA_ds['train'].add_column(name="user_content", column=[
            i['question'] for i in PubMedQA_ds['train']
        ])
        PubMedQA_ds['test'] = PubMedQA_ds['test'].add_column(name="user_content", column=[
            i['question'] for i in PubMedQA_ds['test']
        ])

        sys_prompt =  "Please answer this question with Yes, No or Maybe."
        if number_shot: # few-shot preprocessing 
            sys_prompt += " Here I will give you few examples: \n"
            fs_ds = PubMedQA_ds["train"]
            for i in range(number_shot):
                sys_prompt += f"Question {i+1}: {fs_ds[i]['user_content']}\nAnswer {i+1}: {fs_ds[i]['final_decision']}\n"
            
            sys_prompt += "Now please answer the user's question."

        PubMedQA_ds['train'] = PubMedQA_ds['train'].add_column(name="sys_content", column=[sys_prompt] * len(PubMedQA_ds['train']))
        PubMedQA_ds['test'] = PubMedQA_ds['test'].add_column(name="sys_content", column=[sys_prompt] * len(PubMedQA_ds['test']))

    
    # {'id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name'}
    if 'MedMCQA' in dataset:
        MedMCQA_ds = load_dataset("openlifescienceai/medmcqa", cache_dir=cache_dir) # train, validation 


        # adding user content and system content  
        MedMCQA_ds['train'] = MedMCQA_ds['train'].add_column(name="user_content", column=[
            f"Question: {i['question']}, Choices: A: {i['opa']}, B: {i['opb']}, C: {i['opc']}, D: {i['opd']}" for i in MedMCQA_ds['train']
        ])
        MedMCQA_ds['validation'] = MedMCQA_ds['validation'].add_column(name="user_content", column=[
            f"Question: {i['question']}, Choices: A: {i['opa']}, B: {i['opb']}, C: {i['opc']}, D: {i['opd']}" for i in MedMCQA_ds['validation']
        ])

        sys_prompt = "Please read the question and pick the most suitable choice from A to D, simply answer A, B, C or D."
        if CoT: # Chain of thought processing
            # Note: Currrently, CoT only supports MedMCQA
            sys_prompt += " Here I will give you few examples: \n"

            with open("CoT/MedMCQA_CoT.txt", "r") as f:
                CoT_prompt = f.read()

            sys_prompt += CoT_prompt
            sys_prompt += "Now please answer the user's question."
        elif number_shot: # few-shot preprocessing 
            sys_prompt += " Here I will give you few examples: \n"
            fs_ds = MedMCQA_ds["train"]
            for i in range(number_shot):
                
                sys_prompt += f"Question {i+1}: {fs_ds[i]['user_content']}\nAnswer {i+1}: {chr(fs_ds[i]['cop'] + 65)}\n"
            
            sys_prompt += "Now please answer the user's question."

        
        MedMCQA_ds['train'] = MedMCQA_ds['train'].add_column(name="sys_content", column=[sys_prompt] * len(MedMCQA_ds['train']))
        MedMCQA_ds['validation'] = MedMCQA_ds['validation'].add_column(name="sys_content", column=[sys_prompt] * len(MedMCQA_ds['validation']))

        
        
    # {'question', 'subject', 'choices', 'answer'}
    if 'MMLU_anatomy' in dataset:  
        anatomy_ds = load_dataset("cais/mmlu", "anatomy", cache_dir=cache_dir) # test, validation 
        anatomy_ds = MMLU_formatter(anatomy_ds, number_shot)

    if 'MMLU_biology' in dataset:
        biology_ds = load_dataset("cais/mmlu", "college_biology", cache_dir=cache_dir) # test, validation 
        biology_ds = MMLU_formatter(biology_ds, number_shot)

    if 'MMLU_medicine' in dataset:
        medicine_ds = load_dataset("cais/mmlu", "college_medicine", cache_dir=cache_dir) # test, validation 
        medicine_ds = MMLU_formatter(medicine_ds, number_shot)

    if 'MMLU_clinical' in dataset:
        clinical_ds = load_dataset("cais/mmlu", "clinical_knowledge", cache_dir=cache_dir) # test, validation 
        clinical_ds = MMLU_formatter(clinical_ds, number_shot)

    # {'ID', 'user_query', 'chatgpt_response', 'hallucination', 'hallucination_span'}
    if 'HaluEval' in dataset:
        halu_ds = load_dataset("pminervini/HaluEval", "general", cache_dir=cache_dir)
        
        sys_prompt =  "Look at the following user query and response from chatgpt. Please answer 'yes' if you think the response is a hallucination or 'no' if you think the response is not a hallucination."

        halu_ds['data'] = halu_ds['data'].add_column(name="sys_content", column=[sys_prompt] * len(halu_ds['data']))

        halu_ds['data'] = halu_ds['data'].add_column(name="user_content", column=[
            f"User query: {i['user_query']}\nChatgpt response: {i['chatgpt_response']}" for i in halu_ds['data']])

    return {"PubMedQA": PubMedQA_ds, "MedMCQA": MedMCQA_ds, "MMLU_anatomy": anatomy_ds, "MMLU_biology": biology_ds, "MMLU_medicine": medicine_ds, "MMLU_clinical": clinical_ds, "HaluEval": halu_ds}
