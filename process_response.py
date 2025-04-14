
#import requests
from datasets import load_from_disk, concatenate_datasets
import yaml
import os 
import groq
from openai import OpenAI
import string 
import time
import argparse
from datasets import Dataset
import json 
import re

# load the attributes of datasets
with open('datasets.json', 'r') as f:
    datasets_info = json.load(f)

def remove_punctuation_and_whitespace(input_string):
    # Remove punctuation
    no_punctuation = input_string.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace
    no_whitespace = no_punctuation.replace(" ", "").replace("\n", "").replace("\t", "")
    return no_whitespace

def load_if_exists(dataset_path):
    if os.path.exists(dataset_path):
        dataset = Dataset.from_json(dataset_path)
        return dataset[-1]["id"], dataset
    else: 
        return None, None


def query_llama3(response, user_content, ds_info):
    """Query Llama3-80B-versatile using Groq"""
    #get api key from environment
    client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
    choices = ['A', 'B', 'C', 'D'] if ds_info['type'] == 'multi' else ['yes', 'no', 'maybe']
    examples = f""" 
Example 1:
Question: A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is, Choices: A. No test is required now as her age is below 35 years, B. Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not, C. Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not, D. blood screening at this point of time will clear the exact picture
Response:  \" Reasoning:The best advice to a 29-year-old pregnant woman with a previous child with Down Syndrome is to have a test done as her age is below 35 years (Choice A)\"
Answer: A

Example 2:
Question: Concentration of tropicamide: Choices: A. 0.01, B. 0.02, C. 0.03, D. 0.04
Response: \" The best 2 answer are A and D\"
Answer: A

Example 3:
Question: In a 6-month-old child, thick curd like white patch appears on the buccal mucosa. On rubbing it leaves an erythematous patch. Most likely diagnosis is: Choice: A. Tuberculosis, B. Lichen planus, C. Lupus erythematous, D. Candidiasis
Response: \" D Here is your step-by-step explanation. To answer this question, I will analyze the clinical information given and discuss the possible diagnosis for each choice.\"
Answer: D

    """
    messages = [
    {
        "role": "system", "content": 
f"""Given a pair of a biomedical question and a response, please shorten the response to the list of choices of short answers: {choices} according to its meaning, or None if the response does not mention any of them.
Please answer with the short answer only. Now here are some examples: {examples}"""}, 
{"role": "system", "content": f"""{user_content}
Response: {response}
Answer: """
}]

    ans = None
    retry_count = 0
    # Reprompt answer if short answer does not match
    while (ans not in [x.lower for x in ds_info["choice"]] and ans not in [x.upper() for x in ds_info["choice"]] and retry_count <= 3):
        
        groq_retry_count = 3
        for attempt in range(groq_retry_count):
            try:
                chat_completion = client.chat.completions.create(
                    messages = messages,
                    model = "llama-3.3-70b-versatile"
                    
                )
                break
            except groq.InternalServerError as e:
                if attempt == groq_retry_count - 1:
                    print("Retry not working, exiting.")
                    raise e 
                print(f"Internal Server Error: {e}, proceeding with retry")
                time.sleep(30)

                
        ans = chat_completion.choices[0].message.content
        ans = remove_punctuation_and_whitespace(ans)
        
        retry_count += 1

    if ans.upper() in choices or ans.lower() in choices:
        print(f"Parseable answer: {ans} received")
        return ans

    else:
        print(f"Unparseable answer: {ans} received")
        return ans + "<UPB>" # <UPB> token indicating an unparsable answer, requiring human postprocessing


def query_deepseek(response, user_content, ds_info):
    """Query Deepseek-V3"""
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    choices = ['A', 'B', 'C', 'D'] if ds_info['type'] == 'multi' else ['yes', 'no', 'maybe']
    examples = f""" 
Example 1:
Question: A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is, Choices: A. No test is required now as her age is below 35 years, B. Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not, C. Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not, D. blood screening at this point of time will clear the exact picture
Response:  \" Reasoning:The best advice to a 29-year-old pregnant woman with a previous child with Down Syndrome is to have a test done as her age is below 35 years (Choice A)\"
Answer: A

Example 2:
Question: Concentration of tropicamide: Choices: A. 0.01, B. 0.02, C. 0.03, D. 0.04
Response: \" The best 2 answer are A and D\"
Answer: A

Example 3:
Question: In a 6-month-old child, thick curd like white patch appears on the buccal mucosa. On rubbing it leaves an erythematous patch. Most likely diagnosis is: Choice: A. Tuberculosis, B. Lichen planus, C. Lupus erythematous, D. Candidiasis
Response: \" D Here is your step-by-step explanation. To answer this question, I will analyze the clinical information given and discuss the possible diagnosis for each choice.\"
Answer: D"""

    messages = [
    {
        "role": "system", "content": 
f"""Given a pair of a biomedical question and a response, please shorten the response to the list of choices of short answers: {choices} according to its meaning, or None if the response does not mention any of them.
Please answer with the short answer only. Now here are some examples: {examples}"""}, 
{"role": "user", "content": f"""{user_content}
Response: {response}
Answer: """
}]

    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    stream=False
    )

    ans = response.choices[0].message.content
    print(ans)

    if ans.upper() in choices or ans.lower() in choices:
        print(f"Parseable answer: {ans} received")
        return ans

    else:
        print(f"Unparseable answer: {ans} received")
        return ans + "<UPB>" # <UPB> token indicating an unparsable answer, requiring human postprocessing


def convert_cop_to_int(example):
    example['cop'] = int(example['cop'])
    return example
def dataset_concat(ds, save_path):
    if os.path.exists(save_path):
        prog_data = Dataset.from_json(save_path)
    else:
        prog_data = None
    
    # Handle compatibility
    # if ('cop' in ds.column_names) and (prog_data is not None):
    #     ds = ds.cast(prog_data.features)
    # if ('answer' in ds.column_names) and (prog_data is not None):
    #     ds = ds.cast(prog_data.features)
    if prog_data is not None:
        ds = ds.cast(prog_data.features)
    
    #Concatenate
    if prog_data:
        combined = concatenate_datasets([prog_data, ds])

        
    else:
        combined = ds 

    
    # save to path
    combined.to_json(save_path, lines=False)

def process_example(row, ds_info, self_con, query_model):
    choices = '|'.join(['A', 'B', 'C', 'D'] if ds_info['type'] == 'multi' else ['Yes', 'No', 'Maybe'])
    pattern = f'Answer:\s*({choices})' 
    # print(f"raw response: {row['response']}")
    if self_con:
        k_res = json.loads(row["response"])
        k_ans = ['' for i in range(self_con)]
        # Attempt to parse the examples
        for i in range(self_con):
            # regex matching for the available choice, ignoring case 
            matcher = re.search(pattern, k_res[i], re.IGNORECASE)

            if matcher:
                print("Answer found: ", matcher.group(1))
                k_ans[i] = matcher.group(1)

            else: # no match, move to Llama3 or Deepseekfor further processing 
                if query_model == "Deepseek":
                    print("no match, proceeding with Deepseek.")
                    row["processed_answer"] = query_deepseek(row["response"], row["user_content"], ds_info)
                else:
                    print("no match, proceeding with groq.")
                    row["processed_answer"] = query_llama3(row["response"], row["user_content"], ds_info)
    
        # aggregate the answer
        row["processed_answer"] = json.dumps(k_ans)
        
    
    else:
        matcher = re.search(pattern, row["response"], re.IGNORECASE)

        if matcher:
            print("Answer found: ", matcher.group(1))
            row['processed_answer'] = matcher.group(1)

        else:
            if query_model == "Deepseek":
                print("no match, proceeding with Deepseek.")
                row["processed_answer"] = query_deepseek(row["response"], row["user_content"], ds_info)
            else:
                print("no match, proceeding with groq.")
                row["processed_answer"] = query_llama3(row["response"], row["user_content"], ds_info)
    
    return row

def evaluate(dpath, dname, savedir, self_con, query_model):
    
    dataset = load_from_disk(dpath)
    dataset = dataset.map(lambda example, idx: {"id": idx}, with_indices=True)

    """
    if dataset_name == "MedMCQA":
        dataset = dataset.select(range(1000))
    """
    
    if (savedir == None): # makeup the save directory 
        savedir = f"shortened/{dpath[dpath.find('/')+1:]}.json"

    print(f">Eval>: Saving directory at {savedir}")

    # Check previous progress 
    progress_id, progress_dataset = load_if_exists(savedir)
    if progress_id: 
        start_index = progress_id + 1
    
        if start_index >= len(dataset):
            return None
        
        # dataset = dataset.select(range(start_index, 1000)) if (dataset_name == "MedMCQA") else dataset.select(range(start_index, len(dataset)))
        dataset = dataset.select(range(start_index, len(dataset)))
        print(f">Eval>: Progress found, starting from row {start_index}")
    else:
        start_index = 0
        print(">Eval>: No Progress found, starting fresh.")
    
    chunk_size = 100
    
    # Process each response in the dataset
    local_start = 0
    try:
        while local_start < len(dataset):
            end_index = min(local_start + chunk_size, len(dataset))
            chunk = dataset.select(range(local_start, end_index))

            # process chunk  
            ds_info = datasets_info[dname]     
            processed_chunk = chunk.map(process_example, fn_kwargs={"ds_info": ds_info, "self_con": self_con, "query_model": query_model})
       
            dataset_concat(processed_chunk, savedir)
            print(f">Eval>: rows {start_index + local_start} to {start_index + end_index} saved")
            local_start = end_index 
    except KeyboardInterrupt:
        #dataset_concat(processed_dataset, savedir)
        print(">Eval>: Keyboard Interrupt.")
    except groq.InternalServerError: 
        #dataset_concat(processed_dataset, savedir)
        print(">Eval>: Exception Occured.")

    print(f">Eval>: {dname} answer processing finished and saved at {savedir}.")

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Argument parser for process_response.py")
    
    # Optional -c argument to load from YAML config
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML config file')

    # Arguments for dataset and model if -c is not provided
    parser.add_argument('-d', '--dataset', nargs='+', type=str, help='Dataset paths')
    parser.add_argument('-s', '--savedir', type=str, default=None, help="directory to be saved")
    parser.add_argument("-sc", "--k_self_con", type=int, default=0, help="number of times in self-consistency")
    parser.add_argument("--query_model", type=str, default="Deepseek")

    args = parser.parse_args()

    query_model = args.query_model
    prompt = None
    self_con = False
    
    if args.config:
        with open(args.config, 'r') as file:
            configs = yaml.safe_load(file)
           

        if configs['response']['from_inference']: # use the response from inference.py
            mname = configs['model']
            dnames = set(configs['dataset']['dataset_names'])
            fs = configs['generation']['few_shot']
            cot = configs['generation']['CoT']
            self_con = configs['generation']['k_self_consistency']

            if self_con:
                prompt = 'SC'
            elif cot:
                prompt = 'CoT'
            else:
                prompt = f'{fs}-shot'

            dname_to_dpath = {dname: f'responses/{mname}/{prompt}/{dname}' for i, dname in enumerate(dnames)}

        else:
            dpaths = configs['response']['response_paths']
            dname_to_dpath = {path[path.rfind('/')+1:]: path for path in dpaths}

        savedir = None if configs['response']['shortened_save_path'] is None else configs['response']['shortened_save_path']

    else: 
        dpaths = args.dataset
        savedir = args.savedir 
        self_con = args.k_self_con

        dname_to_dpath = {path[path.rfind('/')+1:]: path for path in dpaths} # dataset name: dataset path 

    for i, dname in enumerate(dname_to_dpath):
        path = dname_to_dpath[dname]
        print(f">Eval>: Processing answers by {path[path.find('/')+1:path.find('/',path.find('/')+1)]} for {dname}")

        evaluate(path, dname, savedir, self_con, query_model)
