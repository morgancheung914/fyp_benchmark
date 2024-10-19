
#import requests
from datasets import load_from_disk
import yaml
from jinja2 import Template
import os 
from groq import Groq

# Function to query the LLaMA3 model
def query_llama3(text):

    #get api key from environment
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    choices = "A, B, C or D"
    choices = "Yes No or Maybe"
    messages = [{"role": "system", "content": f"""
    The user will provide you a sentence that answers a biomedical question, please determine the short answer of the sentence from this list of choices: {choices}
    Please answer with the short answer only.
    """},
    {
        "role": "user", "content": text
    }
    ]
    chat_completion = client.chat.completions.create(
        messages = messages,
        model = "llama3-8b-8192"
    )
    
    print(chat_completion.choices[0].message.content)

    # Make the request
    #response = requests.post(api_endpoint, json=payload, headers=headers)
    #response_data = response.json()

    # Extract the response
    #return response_data['choices'][0]['text'].strip()

def process_example(example):
    example["processed_answer"] = query_llama3(example["response"])

    return example 

def evaluate(dataset_path, dataset_name):
    
    dataset = load_from_disk(dataset_path)

    # Process each response in the dataset
    processed_dataset = dataset.map(process_example)

    # Save the updated dataset
    processed_dataset.save_to_disk(dataset_path)

    print(f">Eval>: {dataset_name} answer processing finished and saved.")


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

  
    for d in list(d_paths.keys()):
        #get dataset name and path

        d_path = d_paths[d]

        evaluate(d_path, d)
        pass

main()