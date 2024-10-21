
#import requests
from datasets import load_from_disk
import yaml
from jinja2 import Template
import os 
from groq import Groq
import string 

def remove_punctuation_and_whitespace(input_string):
    # Remove punctuation
    no_punctuation = input_string.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace
    no_whitespace = no_punctuation.replace(" ", "").replace("\n", "").replace("\t", "")
    return no_whitespace

# Function to query the LLaMA3 model
def query_llama3(text, dataset_name):

    #get api key from environment
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    if dataset_name in ["MMLU_biology", "MMLU_anatomy", "MMLU_medicine", "MMLU_clinical"]:
        choices = "A, B, C or D"
    if dataset_name in ["PubMedQA"]:
        choices = "Yes No or Maybe"
        
    messages = [
    {
        "role": "user", "content": f"""This is a string that answers a biomedical question, please determine the short answer of the sentence from this list of choices: {choices}
    Please answer with the short answer only and nothing else. This is the string: \"""" + text + "\" This is the end of the string."
    }
    
    ]

    ans = None
    retry_count = 0
    # Reprompt answer if short answer does not match

    while (ans not in ["A", "B", "C", "D", "Yes", "No", "Maybe"] and retry_count <= 3):
        chat_completion = client.chat.completions.create(
            messages = messages,
            model = "llama-3.1-70b-versatile"
            
        )
        
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

def process_example(example, dataset_name):
    example["processed_answer"] = query_llama3(example["response"], dataset_name)

    return example 

def evaluate(dataset_path, dataset_name, model_name):
    
    dataset = load_from_disk(dataset_path)

    # Process each response in the dataset
    
    processed_dataset = dataset.map(process_example, fn_kwargs={"dataset_name": dataset_name})

    # Save the updated dataset
    processed_dataset.to_json(f"shortened/{model_name}/{dataset_name}", lines=False)

    print(f">Eval>: {dataset_name} answer processing finished and saved.")

    return


def main():

    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    model_name = configs['model']

    # Create a Jinja2 template from the content
    template = Template(yaml.dump(configs['response']))

    # Render the template with variables
    rendered_content = template.render(model = model_name)

    # Load the rendered YAML
    rendered_config = yaml.safe_load(rendered_content)


    d_paths = rendered_config['chosen_datasets']
  
    for d in list(d_paths.keys()):
        #get dataset name and path
        print(f">Eval>: Processing answers by {model_name} for {d}")
        d_path = d_paths[d]

        evaluate(d_path, d, model_name)

main()