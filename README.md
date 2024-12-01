## Introduction 

This project aims to provide a generalized framwork for evaluating the performance existing LLMs on several medical QA tasks and hallucination detection.

The framework now supports:
- Models: ChatGLM, Internist, Llama3, Med42, Meditron
- Datasets: MMLU_biology, MMLU_anatomy, MMLU_medicine, MMLU_clinical, PubMedQA, MedMCQA, HaluEval

We would love to include more models and datasets in the future. 

## Setup
Install all the required dependencies before getting start.
```
pip install -r requirements.txt
```

## Basic Usage
The framework is divided into three steps. 

### Inference 
First get the raw model responses. 

Please try to run the following script in an environemnt with CUDA-compatible GPU for faster inference. 
```
python3 inference.py
```
Following parameters are provided
- `model` - select one model for generation each time 
- `dataset` - inference on more than one datasets is supported 
- `few_shot` - number of shots prompted into the model
- `CoT` - true if you would like to perform Chain of Thoughts
- `k_self_consistency` - number of Self Consistency prompted into the model
- `top_pv` - top_p of the model
- `temperature` - temperature of the model
- `batch_size` - batch size of the evaluating sets

Please refer to config.yaml for modifications 

### Response process
Then, the raw responses would be shortened by calling llama-3.1-70b-versatile using Groq API to the designated answer space for easier evaluation.

```
export GROQ_API_KEY=your_key_here
python3 process_response.py -c config.yaml
```
Remeber to specify the input path by `chosen_dataset` in config.yaml, you may also want to specify the output path by `shortened_save_path`

or for automatic API key deployment
```
bash ./autoshort_3.sh
```
In the script, replace these few things:
1. groq.txt -> a txt file path that contains your own API keys
2. log_file = -> a log file path 
3. config.yaml -> a yaml file you wish to use on process_response.py

### Evaluation 
Finally, get the accuracy for the processed responses
```
python3 eval.py 
```
Specify the directory of processed responses by `eval` in config.yaml
e.g. `shortened/llama3/0-shot/` for calculating all 0-shot accuracies of the datasets from llama3

Check out https://drive.google.com/drive/folders/1H37kkPxt082KgpfQraPAjrR5PbklFaLQ?usp=drive_link for the raw responses and processed responses generated.
