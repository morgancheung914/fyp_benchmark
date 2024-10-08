from src.llama3 import Llama3, Med42
# TODO: Choosing the models from the config 

model = Med42(None)

model.load_model()
messages = [
    {"role": "system", "content": "You are an intelligent assistant."},
    {"role": "user", "content": "Who are you?"},
]

print(model.predict(messages, max_length = 150, num_return_seq = 1, temperature = 1))
