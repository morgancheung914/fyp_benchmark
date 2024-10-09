##from src.med42 import Med42
#from src.llama3 import Llama3

#from src.internist import Internist
from preprocess import process_data, chat_formatter
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

bench_datasets = set(configs['dataset']['dataset_names'])
#load dataset

datasets_dict = process_data(d = bench_datasets, cache_dir = 'dataset/cache')
print(f">Bench>: Datasets loaded: {configs['dataset']['dataset_names']}")
for dataset in list(datasets_dict.keys()):
    if datasets_dict[dataset] == None:
        continue 
    else:
        datasets_dict[dataset] = chat_formatter(datasets_dict[dataset], name = dataset, tokenizer = model.tokenizer)
print(f">Bench>: Datasets preprocessing finished: {configs['dataset']['dataset_names']}")



#messages = [
    #{"role": "system", "content": "Please answer the question below"},
    #{"role": "user", "content": "Who are you?"},
#]


print(model.predict(datasets_dict['PubMedQA']['test'][0]["token_text"], max_length = 1000, num_return_seq = 1, temperature = 1.5))




