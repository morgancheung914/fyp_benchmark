##from src.med42 import Med42
#from src.llama3 import Llama3

#from src.internist import Internist
import os
from preprocess import process_data
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm 


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

            
        else:   
            selected_ds = datasets_dict[dataset]['test']

        print(f">Bench>: Datasets preprocessing for {dataset} finished.")

        #test for deterministicness

        ds_test = [[
                {"role": "system", "content": i['sys_content']},
                
                {"role": "user", "content": i['user_content']}] for i in selected_ds]

        dataloader = DataLoader(ds_test, batch_size = 3, shuffle=False, collate_fn = lambda x: x)
        responses = []
        print(f">Bench>: Starting inference on {dataset}.")

        for batch in tqdm(dataloader):
            #print(batch)
            batch_responses = (model.batch_predict(batch, max_length = 300, num_return_seq = 1, temperature = 1, top_p = 0.9))
            #print(batch_responses)
            print(batch_responses)

            responses.extend(batch_responses) 
        selected_ds = selected_ds.add_column(name = "response", column = responses)
        if CoT:
            selected_ds.save_to_disk(f'responses/{configs["model"]}/CoT/{dataset}_1000')

        elif few_shot:
            selected_ds.save_to_disk(f'responses/{configs["model"]}/{number_shot}-shot/{dataset}_1000')


        print(f">Bench>: Inferencing on {dataset} finished.")



