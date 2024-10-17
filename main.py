##from src.med42 import Med42
#from src.llama3 import Llama3

#from src.internist import Internist
import os
from preprocess import process_data
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader
import yaml


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

# load datasets, the returned dict stores the processed datasetDict w.r.t each dataset 
datasets_dict = process_data(bench_datasets, 'dataset/cache')
print(f">Bench>: Datasets loaded: {configs['dataset']['dataset_names']}")



# inference 
for dataset in list(datasets_dict.keys()):
    if datasets_dict[dataset] == None:
        if dataset in bench_datasets:
            print("Dataset processing error")

        else:
            continue 

    else:
        # parse the contents into the designated prompt template 
        if dataset == 'MedMCQA':
            ds_test = [[{"role": "system", "content": i['sys_content']},
                {"role": "user", "content": i['user_content']}] for i in datasets_dict[dataset]['validation']]
        else:   
            ds_test = [[{"role": "system", "content": i['sys_content']},
                {"role": "user", "content": i['user_content']}] for i in datasets_dict[dataset]['test']]
        
        dataloader = DataLoader(ds_test, batch_size = 3, shuffle=False, collate_fn = lambda x: x)

        for batch in dataloader:
            print(model.batch_predict(batch, max_length = 50, num_return_seq = 1, temperature = 1))








