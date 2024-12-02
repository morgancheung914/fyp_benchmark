
#import requests
from datasets import load_from_disk
from datasets import concatenate_datasets
import yaml
from jinja2 import Template
import os 
from groq import Groq
import groq
import string 
import time
import argparse
from datasets import Dataset
import json 
import re



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

# Function to query the LLaMA3 model
def query_llama3(text, dataset_name):

    #get api key from environment
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    if dataset_name in ["MMLU_biology", "MMLU_anatomy", "MMLU_medicine", "MMLU_clinical", "MedMCQA"]:
        choices = "A, B, C or D"
    if dataset_name in ["PubMedQA"]:
        choices = "Yes No or Maybe"
    if dataset_name in ["HaluEval"]:
        choices = "Yes or No"
    
    examples = """ String 1:  \"a
Reasoning:The best advice to a 29-year-old pregnant woman with a previous child with Down Syndrome is to have a test done as her age is below 35 years (Choice A). This is\"
Answer 1: A

String 2: \"c

            The best 2 answer are A and D.

            Question: Concentration of tropicamide:, Choices: A: 0.01, B: 0.02, C\"
Answer 2: C

String 3: \" d

Here is your step-by-step explanation

To answer this question, I will analyze the clinical information given and discuss the possible diagnosis for each choice.

1. Tuberculosis (Ans. A): T \"

Answer 3: D

    """
    messages = [
    {
        "role": "user", "content": f"""This is a string that answers a biomedical question, please determine the short answer of the sentence from this list of choices: {choices}, or None if the long answer does not mention any of the choices, usually the choice will be at the start of the string, they could be in upper-case or lower-case
    Please answer with the short answer only and nothing else. Now here are some examples: {examples}

    
    And here is the string you need to determine: \"""" + text + "\" This is the end of the string."
    }
    
    ]

    ans = None
    retry_count = 0
    # Reprompt answer if short answer does not match
    print(f"text: {text}\n")

    while (ans not in ["A", "B", "C", "D", "Yes", "No", "Maybe"] and retry_count <= 3):
        
        groq_retry_count = 3
        for attempt in range(groq_retry_count):
            try:
                chat_completion = client.chat.completions.create(
                    messages = messages,
                    model = "llama-3.1-70b-versatile"
                    
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

    if ans in ["A", "B", "C", "D", "Yes", "No", "Maybe"]:
        print(f"Parseable answer: {ans} received")
        return ans

    else:
        print(f"Unparseable answer: {ans} received")
        return ans + "<UPB>"
    

    # Make the request
    #response = requests.post(api_endpoint, json=payload, headers=headers)
    #response_data = response.json()

    # Extract the response
    #return response_data['choices'][0]['text'].strip()
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

    
    #Save to path
    combined.to_json(save_path, lines=False)

def process_example(example, dataset_name, self_con):
    if self_con:
        k_paths = json.loads(example["response"])
        k_ans = ['' for i in range(self_con)]
        # Attempt to parse the examples
        for i in range(self_con):
            # regex matching
            pattern = r'\"Answer\":\s*([ABCD])'

            matcher = re.search(pattern, k_paths[i])

            if matcher:
                print("Answer found: ", matcher.group(1))
                k_ans[i] = matcher.group(1)

            else:
                print("no match, proceeding with groq.")
                k_ans[i] = query_llama3(example["response"], dataset_name)
    
        #aggregate the answer
        example["processed_answer"] = json.dumps(k_ans)
        
    
    else:
        example["processed_answer"] = query_llama3(example["response"], dataset_name)
    
    return example 

def evaluate(dataset_path, dataset_name, savedir, self_con):
    
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(lambda example, idx: {"id": idx}, with_indices=True)
    if dataset_name == "MedMCQA":
        dataset = dataset.select(range(1000))

    # Check previous progress 

    if (savedir == None): # makeup the save directory 
        savedir = f"shortened/{dataset_path[10:]}"

    print(f">Eval>: Saving directory at {savedir}")
    progress_id, progress_dataset = load_if_exists(savedir)
    if progress_id: 
        start_index = progress_id + 1
    
        if start_index >= len(dataset):
            return None
        dataset = dataset.select(range(start_index, 1000)) if (dataset_name == "MedMCQA") else dataset.select(range(start_index, len(dataset)))
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

            ## process chunk          
            processed_chunk = chunk.map(process_example, fn_kwargs={"dataset_name": dataset_name, "self_con": self_con})
            
            #processed_dataset = dataset.map(process_example, fn_kwargs={"dataset_name": dataset_name})
            print(processed_chunk[0])
            # Save the updated dataset
            dataset_concat(processed_chunk, savedir)
            print(f">Eval>: rows {start_index + local_start} to {start_index + end_index} saved")
            local_start = end_index 
    except KeyboardInterrupt:
        #dataset_concat(processed_dataset, savedir)
        print(">Eval>: Keyboard Interrupt.")
    except groq.InternalServerError: 
        #dataset_concat(processed_dataset, savedir)
        print(">Eval>: Exception Occured.")
    

    # Concatenate progress with new work 
    
    

    
    # if savedir:
    #     processed_dataset.to_json(f"{savedir}", lines=False)
    
    # else:
    #     if few_shot: 
    #         processed_dataset.to_json(f"shortened/{model_name}/{dataset_name}_fsp", lines=False)
    #     else:
            
    #         processed_dataset.to_json(f"shortened/{model_name}/{dataset_name}", lines=False)

    print(f">Eval>: {dataset_name} answer processing finished and saved at {savedir}.")

    return


def main():

    parser = argparse.ArgumentParser(description="Argument parser for process_response.py")
    
    # Optional -c argument to load from YAML config
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML config file')

    # Arguments for dataset and model if -c is not provided
    parser.add_argument('-d', '--dataset', nargs='+', type=str, help='Dataset paths')
    parser.add_argument('-s', '--savedir', type=str, default=None, help="directory to be saved")
    parser.add_argument("-sc", "--k_self_con", type=int, default=0, help="number of times in self-consistency")

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as file:
            configs = yaml.safe_load(file)

        if configs['response']['from_inference']: # decide if to use the model and parameters in inference
            model_name = configs['model']
            fs = configs['generation']['few_shot']
            cot = configs['generation']['CoT']
            self_con = configs['generation']['k_self_consistency']

            if self_con:
                prompt = 'SC'
            elif cot:
                prompt = 'CoT'
            else:
                prompt = f'{fs}-shot'

        savedir = None if configs['response']['shortened_save_path'] is None else configs['response']['shortened_save_path']

        # Create a Jinja2 template from the content
        template_content = yaml.dump({
            'dataset': configs['dataset'],
            'model': configs['model'],
            'generation': configs['generation'],
            'response': configs['response'],
            'eval': configs['eval']
        })
        template = Template(template_content)

        # Render the template with variables
        rendered_content = template.render(model = model_name, prompt = prompt)

        # Load the rendered YAML
        rendered_config = yaml.safe_load(rendered_content)
        
        d_paths = rendered_config['response']['chosen_datasets']
    else: 
        dp = args.dataset
        savedir = args.savedir # currently only support one processing one dataset at one script instance
        self_con = args.k_self_con

        d_paths = {p[len(p)-p[::-1].index('/'):]: p for p in dp}
        

    for d in list(d_paths.keys()):
        # get dataset name and path
        d_path = d_paths[d]
        print(f">Eval>: Processing answers by {d_path[10:10+d_path[10:].index('/')]} for {d}")
        
        evaluate(d_path, d, savedir, self_con)

main()