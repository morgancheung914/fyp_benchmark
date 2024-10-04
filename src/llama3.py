import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
cache_dir = "./models"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": cache_dir}, device_map="auto"
)

system_prompt = "You are a medical chatbot answering user inquries."
user_msg_1 = "Hello, I have rashes on my skin and I am feeling a headache today. Please give me a diagnosis."

prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg_1} <|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

print(pipeline(prompt, max_new_tokens=200, temperature=1))


