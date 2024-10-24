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

# load datasets, the returned dict stores the processed datasetDict w.r.t each dataset 
datasets_dict = process_data(bench_datasets, 'dataset/cache')
print(f">Bench>: Datasets loaded: {configs['dataset']['dataset_names']}")

few_shot = configs['generation']['few_shot']

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
            selected_ds = datasets_dict[dataset]['validation']

            if few_shot:
                fs_ds = datasets_dict[dataset]["train"]
                class_map = {0: "a", 1: "b", 2: "c", 3: "d"}
                
                few_shot_prompt = f""" Here I will give you a few examples:
                Question 1: {fs_ds[0]['user_content']}
                Answer 1: {class_map[fs_ds[0]["cop"]]}

                Question 2: {fs_ds[1]['user_content']}
                Answer 2: {class_map[fs_ds[1]["cop"]]}

                Question 3: {fs_ds[2]['user_content']}
                Answer 3: {class_map[fs_ds[2]["cop"]]}

                Now please answer the user's question:
                """
                ds_test = [[
                    {"role": "system", "content": i['sys_content'] + few_shot_prompt},
                    
                    {"role": "user", "content": i['user_content']}] for i in selected_ds]
            else:

                ds_test = [[{"role": "system", "content": i['sys_content']},
                    {"role": "user", "content": i['user_content']}] for i in selected_ds]
        else:   
            selected_ds = datasets_dict[dataset]['test']
            ds_test = [[{"role": "system", "content": i['sys_content']},
                {"role": "user", "content": i['user_content']}] for i in selected_ds]
        
        print(f">Bench>: Datasets preprocessing for {dataset} finished.")

        #test for deterministicness

        dataloader = DataLoader(ds_test, batch_size = 3, shuffle=False, collate_fn = lambda x: x)
        responses = []
        print(f">Bench>: Starting inference on {dataset}.")

        for batch in tqdm(dataloader):
            #print(batch)
            batch_responses = (model.batch_predict(batch, max_length = 50, num_return_seq = 1, temperature = 1, top_p = 0.9))
            #print(batch_responses)
            responses.extend(batch_responses)
        selected_ds = selected_ds.add_column(name = "response", column = responses)
        
        selected_ds.save_to_disk(f'responses/{configs["model"]}/{dataset}')


        print(f">Bench>: Inferencing on {dataset} finished.")



