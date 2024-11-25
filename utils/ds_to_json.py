from datasets import load_from_disk

# Load your dataset from disk
dataset = load_from_disk('/Users/morgan/Documents/Study/AIST FYP/fyp_benchmark/tmp/ChatGLM_CoT')

# Save the dataset to a JSON file
dataset.to_json('/Users/morgan/Documents/Study/AIST FYP/fyp_benchmark/shortened/ChatGLM/cot/MedMCQA_1000.json', lines=False)
