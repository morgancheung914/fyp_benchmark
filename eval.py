
#import requests
from datasets import load_from_disk
import yaml
from jinja2 import Template
import os 
from groq import Groq

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
    
    chat_completion = client.chat.completions.create(
        messages = messages,
        model = "llama-3.1-70b-versatile"
        
    )
    
    return chat_completion.choices[0].message.content


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
    processed_dataset.to_json(f"shortened/{model_name}/{dataset_name}")

    print(f">Eval>: {dataset_name} answer processing finished and saved.")


    return

def main():

    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    model_name = configs['model']

    # Create a Jinja2 template from the content
    template = Template(yaml.dump(configs['evaluate']))

    # Render the template with variables
    rendered_content = template.render(model = model_name)

    # Load the rendered YAML
    rendered_config = yaml.safe_load(rendered_content)

    
    

    d_paths = rendered_config['chosen_datasets']
    print(d_paths)
  
    for d in list(d_paths.keys()):
        #get dataset name and path
        print(d_paths)
        d_path = d_paths[d]

        evaluate(d_path, d, model_name)
        

main()