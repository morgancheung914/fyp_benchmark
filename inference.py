##from src.med42 import Med42
#from src.llama3 import Llama3

#from src.internist import Internist
import os
from preprocess import process_data
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm 
import json
import argparse


parser = argparse.ArgumentParser(description="Argument parser for inference.py")
    
# Optional -c argument to load from YAML config
parser.add_argument('-c', '--config', type=str, default=None, help='Path to the YAML config file')

# Arguments for dataset and model if -c is not provided
parser.add_argument('-d', '--dataset', nargs='+', type=str, help='List of dataset paths')
parser.add_argument('-m', '--model', type=str, help='Model name or path')
parser.add_argument("-f", "--fewshot", type=int, default=0, help="Enable few-shot mode")
parser.add_argument('-t', '--cot', type=bool, default=False, help="Enable Chain of Thoughts")
parser.add_argument("-s", "--k_self_con", type=int, default=0, help="Number of times in self-consistency")
parser.add_argument("-p", "--top_p", type=float, default=0.9, help="Generation top_p")
parser.add_argument("-r", "--temperature", type=float, default=1.0, help="Generation temperature")
parser.add_argument("-b", "--batch_size", type=int, default=3, help="Generation batch size")

args = parser.parse_args()
if args.config:
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    model_file = __import__(f'''src.{configs['model']}''', fromlist=[configs['model']+"Model"])
    model_chosen = getattr(model_file, configs['model']+"Model")
    # get the required datasets from config 
    bench_datasets = set(configs['dataset']['dataset_names'])

    # get the generation parameters 
    number_shot = configs['generation']['few_shot']
    CoT = configs['generation']['CoT']
    self_cons = configs['generation']['k_self_consistency']
    top_p = configs['generation']['top_p']
    temp = configs['generation']['temperature']
    batch_size = configs['generation']['batch_size']

else:
    model_name = args.model
    bench_datasets = set(args.dataset)
    number_shot = args.fewshot
    CoT = args.cot
    self_cons = args.k_self_con
    top_p = args.top_p
    temp = args.temperature
    batch_size = args.batch_size

    model_file = __import__(f'''src.{model_name}''', fromlist=[model_name+"Model"])
    model_chosen = getattr(model_file, model_name+"Model")

# Choosing the models from the config 
model = model_chosen(None)
model.load_model()
# Enforce CoT if self_cons is set to true
if self_cons:
    CoT = True

# load datasets, the returned dict stores the processed datasetDict w.r.t each dataset 
datasets_dict = process_data(bench_datasets, cache_dir='dataset/cache', number_shot=number_shot, CoT=CoT)
print(f">Bench>: Datasets loaded: {configs['dataset']['dataset_names']}")


# inference 
for dataset in list(datasets_dict.keys()):
    if dataset in bench_datasets and datasets_dict[dataset] == None:
        print(f">Error>: Error encountered when processing {dataset}")

    else:
        # parse the contents into the designated prompt template 
        if dataset == 'MedMCQA':
            selected_ds = datasets_dict[dataset]['validation']
            selected_ds = selected_ds.select(range(1000))
            print(selected_ds[0])

        elif dataset == 'HaluEval':
            selected_ds = datasets_dict[dataset]['data']
            selected_ds = selected_ds.select(range(500))
            
        else:   
            selected_ds = datasets_dict[dataset]['test']

        print(f">Bench>: Datasets preprocessing for {dataset} finished.")

        #test for deterministicness

        ds_test = [[
                {"role": "system", "content": i['sys_content']},
                
                {"role": "user", "content": i['user_content']}] for i in selected_ds]

        dataloader = DataLoader(ds_test, batch_size = batch_size, shuffle=False, collate_fn = lambda x: x)
        responses = []
        print(f">Bench>: Starting inference on {dataset}.")

        for batch in tqdm(dataloader):
            #print(batch)
            num_seq = 1 if not self_cons else self_cons #set number of sequences to generate
  
            batch_responses = (model.batch_predict(batch, max_length = 300, num_return_seq = num_seq, temperature = temp, top_p = top_p))

            #print(f"batch: {batch_responses}\n")
            #print(f"len: {len(batch_responses)}")

            if self_cons: # if self_consistency is in effect, divide the list of results with the # of self-cons generations
                for i in range(1):
                    curr = batch_responses[i*self_cons: i*self_cons + self_cons]
                    #pack as json string
                    curr_json = json.dumps(curr)
                    responses.append(curr_json)
                    #print(responses)

            else:
                #self_consistency is not in effect
                responses.extend(batch_responses) 
        
        # appending the model response
        selected_ds = selected_ds.add_column(name = "response", column = responses)

        if self_cons != False: #self_consistency responses
            selected_ds.save_to_disk(f'responses/{configs["model"]}/SC/{dataset}')

        elif CoT: # CoT responses
            selected_ds.save_to_disk(f'responses/{configs["model"]}/CoT/{dataset}')

        else: # 0-shot, 1-shot, 3-shot reponses
            selected_ds.save_to_disk(f'responses/{configs["model"]}/{number_shot}-shot/{dataset}')


        print(f">Bench>: Inferencing on {dataset} finished.")



