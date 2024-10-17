from datasets import load_dataset
import argparse 
import re 

def process_data(dataset, cache_dir): # takes in dataset and cache_dir configs 
    def MMLU_formatter(ds):
        # adding system content and user content 
        ds['test'] = ds['test'].add_column(name="sys_content", column=[
            "Please read the question and pick the most suitable choice from A to D, simply answer A, B, C or D"] * len(ds['test']))
        ds['test'] = ds['test'].add_column(name="user_content", column=[
            f"Question: {i['question']}, Choices: A: {i['choices'][0]}, B: {i['choices'][1]}, C: {i['choices'][2]}, D: {i['choices'][3]}" for i in ds['test']])
         
        return ds 

    PubMedQA_ds, MedMCQA_ds, anatomy_ds, biology_ds, medicine_ds, clinical_ds = None, None, None, None, None, None
    

    # {'pub_id', 'question', 'context', 'long_answer', 'final_decision'}
    if 'PubMedQA' in dataset:
        PubMedQA_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train', cache_dir=cache_dir)
        PubMedQA_ds = PubMedQA_ds.train_test_split(test_size=0.5) # train, test 
        
        # adding system content and user content 
        PubMedQA_ds['train'] = PubMedQA_ds['train'].add_column(name="sys_content", column=[
            "Please answer this question with Yes, No or Maybe."] * len(PubMedQA_ds['train']))
        PubMedQA_ds['test'] = PubMedQA_ds['test'].add_column(name="sys_content", column=[
            "Please answer this question with Yes, No or Maybe."] * len(PubMedQA_ds['test']))

        PubMedQA_ds['train'] = PubMedQA_ds['train'].add_column(name="user_content", column=[
            i['question'] for i in PubMedQA_ds['train']
        ])
        PubMedQA_ds['test'] = PubMedQA_ds['test'].add_column(name="user_content", column=[
            i['question'] for i in PubMedQA_ds['test']
        ])

    
    # {'id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name'}
    if 'MedMCQA' in dataset:
        MedMCQA_ds = load_dataset("openlifescienceai/medmcqa", cache=cache_dir) # train, validation 

        # adding system content and user content 
        MedMCQA_ds['train'] = MedMCQA_ds['train'].add_column(name="sys_content", column=[
            "Please read the question and pick the most suitable choice from A to D, simply answer A, B, C or D"] * len(MedMCQA_ds['train']))
        MedMCQA_ds['validation'] = MedMCQA_ds['validation'].add_column(name="sys_content", column=[
            "Please read the question and pick the most suitable choice from A to D, simply answer A, B, C or D"] * len(MedMCQA_ds['validation']))

        MedMCQA_ds['train'] = MedMCQA_ds['train'].add_column(name="user_content", column=[
            f"Question: {i['question']}, Choices: A: {i['opa']}, B: {i['opb']}, C: {i['opc']}, D: {i['opd']}" for i in MedMCQA_ds['train']
        ])
        MedMCQA_ds['validation'] = MedMCQA_ds['validation'].add_column(name="user_content", column=[
            f"Question: {i['question']}, Choices: A: {i['opa']}, B: {i['opb']}, C: {i['opc']}, D: {i['opd']}" for i in MedMCQA_ds['validation']
        ])


    # {'question', 'subject', 'choices', 'answer'}
    if 'MMLU_anatomy' in dataset:  
        anatomy_ds = load_dataset("cais/mmlu", "anatomy", cache_dir=cache_dir) # test, validation 
        anatomy_ds = MMLU_formatter(anatomy_ds)

    if 'MMLU_biology' in dataset:
        biology_ds = load_dataset("cais/mmlu", "college_biology", cache_dir=cache_dir) # test, validation 
        biology_ds = MMLU_formatter(biology_ds)

    if 'MMLU_medicine' in dataset:
        medicine_ds = load_dataset("cais/mmlu", "college_medicine", cache_dir=cache_dir) # test, validation 
        medicine_ds = MMLU_formatter(medicine_ds)

    if 'MMLU_clinical' in dataset:
        clinical_ds = load_dataset("cais/mmlu", "clinical_knowledge", cache_dir=cache_dir) # test, validation 
        clinical_ds = MMLU_formatter(clinical_ds)

    return {"PubMedQA": PubMedQA_ds, "MedMCQA_ds": MedMCQA_ds, "MMLU_anatomy": anatomy_ds, "MMLU_biology": biology_ds, "MMLU_medicine": medicine_ds, "MMLU_clinical": clinical_ds}
