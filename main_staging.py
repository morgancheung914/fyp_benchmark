from src.med42 import Med42
from src.llama3 import Llama3

# TODO: Choosing the models from the config 

model = Med42(None)

model.load_model()

#load dataset

messages = [
    {"role": "system", "content": "Please answer the question below"},
    {"role": "user", "content": "Who are you?"},
]

print(model.predict(messages, max_length = 1000, num_return_seq = 1, temperature = 1.5))




