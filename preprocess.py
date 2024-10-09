from datasets import load_dataset
import argparse 
import re 

def process_data(d, cache_dir): # takes in dataset and cache_dir configs 
    dataset = set(d)
    
    PubMedQA_ds, MedMCQA_ds, anatomy_ds, biology_ds, medicine_ds, clinical_ds = None, None, None, None, None, None
    
    
    # {'pub_id', 'question', 'context', 'long_answer', 'final_decision'}
    if 'PubMedQA' in dataset:
        PubMedQA_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train', cache_dir=cache_dir)
        PubMedQA_ds = PubMedQA_ds.train_test_split(test_size=0.5) # train, test 
    
    

    # {'id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name'}
    if 'MedMCQA' in dataset:
        MedMCQA_ds = load_dataset("openlifescienceai/medmcqa") # train, test, validation 

    # {'question', 'subject', 'choices', 'answer'}
    if 'MMLU_anatomy' in dataset:  
        anatomy_ds = load_dataset("cais/mmlu", "anatomy", cache_dir=cache_dir) # test, validation 

    if 'MMLU_biology' in dataset:
        biology_ds = load_dataset("cais/mmlu", "college_biology", cache_dir=cache_dir) # test, validation 

    if 'MMLU_medicine' in dataset:
        medicine_ds = load_dataset("cais/mmlu", "college_medicine", cache_dir=cache_dir) # test, validation 

    if 'MMLU_clinical' in dataset:
        clinical_ds = load_dataset("cais/mmlu", "clinical_knowledge", cache_dir=cache_dir) # test, validation 

    ret_dict = {"PubMedQA": PubMedQA_ds, "MedMCQA_ds": MedMCQA_ds, "MMLU_anatomy": anatomy_ds, "MMLU_biology": biology_ds, "MMLU_medicine": medicine_ds, "MMLU_clinical": clinical_ds}
    return ret_dict



    

def format_chat_template(row, qcol, system_prompt, tokenizer):
    row_json = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": row[qcol]}]
    row["token_text"] = row_json
    return row

def format_MedMcQa(row, tokenizer):
    row_json = [{"role": "system", "content": "Please read the question and pick the most suitable choice from A to D, simply answer A, B, C or D"},
                {"role": "user", "content": "Question: " + row["question"] + "Choices: " + f""" A: {row["opa"]}, B: {row["opb"]}, C: {row["opc"]}, D: {row["opd"]}"""},
              ]
    row["token_text"] = row_json
    return row

def format_MMLU(row, tokenizer):

    row_json = [{"role": "system", "content": "Please read the question and pick the most suitable choice from A to D, simply answer A, B, C or D"},
                {"role": "user", "content": "Question: " + row['question'] + "Choices: " + f"A: {row['choices'][0]}, B: {row['choices'][1]}, C: {row['choices'][2]}, D: {row['choices'][3]}"},
                ]

    row["token_text"] = row_json
    return row


def chat_formatter(dataset, name = None, question='question', system_prompt = "Please answer this question with Yes, No or Maybe.", tokenizer = None):
    
    if name == "MedMCQA":
        dataset = dataset.map(
        format_MedMcQa,
        fn_kwargs = {"tokenizer": tokenizer},
        num_proc=4,
    )
    elif name == "PubMedQA":
        dataset = dataset.map(
            format_chat_template,
            fn_kwargs = {'qcol': question, 'system_prompt': system_prompt, "tokenizer": tokenizer},
            num_proc=4,
        )
    else:
        dataset = dataset.map(format_MMLU, fn_kwargs = {"tokenizer": tokenizer}, num_proc=4)
    return dataset
