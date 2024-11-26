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

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)


# Choosing the models from the config 

model_file = __import__(f'''src.{configs['model']}''', fromlist=[configs['model']+"Model"])
model_chosen = getattr(model_file, configs['model']+"Model")
#model = Llama3(None)
#model = Internist(None)
model = model_chosen(None)
model.load_model()


# get the required datasets from config 
bench_datasets = set(configs['dataset']['dataset_names'])
number_shot = configs['generation']['few_shot']
CoT = configs['generation']['CoT']
self_cons = configs['generation']['k_self_consistency']

#Enforce CoT if self_cons is set to true
if self_cons != False:
    CoT = True

# load datasets, the returned dict stores the processed datasetDict w.r.t each dataset 
datasets_dict = process_data(bench_datasets, 'dataset/cache', number_shot, CoT)
print(f">Bench>: Datasets loaded: {configs['dataset']['dataset_names']}")


# inference 
for dataset in list(datasets_dict.keys()):
    if datasets_dict[dataset] == None:
        if dataset in bench_datasets:
            print(f">Error>: Error encountered when processing {dataset}")

        else:
            continue 

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

        dataloader = DataLoader(ds_test, batch_size = 1, shuffle=False, collate_fn = lambda x: x)
        responses = []
        print(f">Bench>: Starting inference on {dataset}.")

        for batch in tqdm(dataloader):
            #print(batch)
            num_seq = 1 if self_cons == False else self_cons #set number of sequences to generate
            print(num_seq)
            batch_responses = (model.batch_predict(batch, max_length = 300, num_return_seq = num_seq, temperature = 1, top_p = 0.9))
            #print(batch_responses)
            print(f"batch: {batch_responses}\n")
            print(f"len: {len(batch_responses)}")

            if self_cons != False: # if self_consistency is in effect, divide the list of results with the # of self-cons generations
                for i in range(1):
                    curr = batch_responses[i*self_cons: i*self_cons + self_cons]
                    #pack as json string
                    curr_json =json.dumps(curr)
                    responses.append(curr_json)
                    print(responses)

            else:
                #self_consistency is not in effect
                responses.extend(batch_responses) 
        selected_ds = selected_ds.add_column(name = "response", column = responses)

        if self_cons != False: #self_consistency responses
            selected_ds.save_to_disk(f'responses/{configs["model"]}/SC/{dataset}')
        if CoT: # CoT responses
            selected_ds.save_to_disk(f'responses/{configs["model"]}/CoT/{dataset}')

        else: # 0-shot, 1-shot, 3-shot reponses
            selected_ds.save_to_disk(f'responses/{configs["model"]}/{number_shot}-shot/{dataset}')


        print(f">Bench>: Inferencing on {dataset} finished.")



