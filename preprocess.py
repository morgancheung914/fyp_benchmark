from datasets import load_dataset
import argparse 
import re 

def preprocess(d, c): # takes in dataset and cache_dir configs 
    dataset = set([i for i in re.split(',| ', d)])
    
    PubMedQA_ds, MedMCQA_ds, anatomy_ds, biology_ds, medicine_ds, clinical_ds = None, None, None, None, None, None

    
    # {'pub_id', 'question', 'context', 'long_answer', 'final_decision'}
    if 'PubMedQA' in dataset or 'all' in dataset:
        PubMedQA_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train', cache_dir=c)
        PubMedQA_ds = PubMedQA_ds.train_test_split(test_size=0.5) # train, test 

    # {'id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name'}
    if 'MedMCQA' in dataset or 'all' in dataset:
        MedMCQA_ds = load_dataset("openlifescienceai/medmcqa") # train, test, validation 

    # {'question', 'subject', 'choices', 'answer'}
    if 'MMLU_anatomy' in dataset or 'all' in dataset:  
        anatomy_ds = load_dataset("cais/mmlu", "anatomy", cache_dir=c) # test, validation 

    if 'MMLU_biology' in dataset or 'all' in dataset:
        biology_ds = load_dataset("cais/mmlu", "college_biology", cache_dir=c) # test, validation 

    if 'MMLU_medicine' in dataset or 'all' in dataset:
        medicine_ds = load_dataset("cais/mmlu", "college_medicine", cache_dir=c) # test, validation 

    if 'MMLU_clinical' in dataset or 'all' in dataset:
        clinical_ds = load_dataset("cais/mmlu", "clinical_knowledge", cache_dir=c) # test, validation 


    return PubMedQA_ds, MedMCQA_ds, anatomy_ds, biology_ds, medicine_ds, clinical_ds

