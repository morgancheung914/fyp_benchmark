import gradio as gr
import subprocess
import sys
import os
from datasets import load_from_disk
import yaml
import json
import shutil
import tempfile
import json 


# load the attributes of datasets
with open('datasets.json', 'r') as f:
    datasets_info = json.load(f)

# currently supported models, datasets, prompting techniques 
models = ['ChatGLM', 'Internist', 'Llama3', 'Med42', 'Meditron']
datasets = ['MMLU_biology', 'MMLU_anatomy', 'MMLU_medicine', 'MMLU_clinical', 'PubMedQA', 'MedMCQA', 'HaluEval']
prompting_techniques = ['Few-shots prompting', 'Chain of Thoughts', 'Self Consistency']

custom_css = """
<style>
#auto_button{
    width: 10px;
    height: 50px;
    font-size: 16px;
    margin-left: auto;
    margin-right: 0;
}

#human_button{
    width: 10px;
    height: 50px;
    font-size: 16px;
    margin-left: 0;
    margin-right: auto;
}

#return_button{   
    width: 60px;
    height: 50px;
    font-size: 16px;
}

#prevEmpty_button,{
    width: 10px;
    height: 50px;
    font-size: 16px;
    margin-left: 0;
}

#prev_button{
    width: 10px;
    height: 50px;
    font-size: 16px;
    margin-left: 0;
}

#next_button{
    width: 10px;
    height: 50px;
    margin-left: auto;
    margin-right: 0;
}

#nextEmpty_button{
    width: 10px;
    height: 50px;
    margin-left: 0;
    margin-right: 0;
}

#save_button{
    width: 30px;
    height: 50px;
    margin-left: auto;
}
</style>
"""    

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)

def get_ds_info(dname):
    global processing_dataset, question_col, answer_col, choice_col, choices 
    ds_info = datasets_info[dname]
    question_col = ds_info['question_col']
    answer_col = ds_info['answer_col']
    choice_col = ds_info['choice_col']
    choices = ds_info['choices']

def load_dataset(load_dir, save_dir):
    # Always load the original dataset from load_dir
    processing_dataset = load_from_disk(load_dir)
    dname = load_dir[load_dir.rindex('/')+1:]
    print(f"Loaded original dataset: {dname} from {load_dir}")

    # get the name of the columns of datasets 
    get_ds_info(dname)

    # Add 'index' column if it doesn't exist
    if 'index' not in processing_dataset.column_names:
        processing_dataset = processing_dataset.add_column('index', list(range(len(processing_dataset))))

    # Initialize 'processed_answer' column if it doesn't exist
    if 'processed_answer' not in processing_dataset.column_names:
        processing_dataset = processing_dataset.add_column('processed_answer', [''] * len(processing_dataset))

    processed_answers = {}
    if os.path.exists(save_dir) and os.listdir(save_dir):
        # Load the saved dataset to get existing processed answers
        saved_dataset = load_from_disk(save_dir)
        print(f"Loaded saved dataset from {save_dir}")

        # Create a mapping from index to processed_answer
        processed_answers = {
            example['index']: example['processed_answer']
            for example in saved_dataset
            if example['processed_answer']
        }

        # Find the index of the last processed entry
        if processed_answers:
            start_idx = max(processed_answers.keys()) + 1
    else:
        start_idx = 0

    return processed_answers, start_idx 

def find_unprocessed_index(processing_dataset, processed_answers, start=0, prev=False):
    if prev: # Find the index of the previous entry wiuhtout a 'processed_answer'
        for i in range(start, -1, -1):
            if i not in processed_answers:
                return i
        return -1
    else: # Find the index of the next entry without a 'processed_answer'
        for i in range(start, len(processing_dataset)):
            if i not in processed_answers:
                return i
        return len(processing_dataset)  # Return length if all entries are processed

def get_row(processing_dataset, index):
    if index >= len(processing_dataset):
        return "You have reached the end of the dataset."
    # Retrieve index, question, and response for the current index
    row = processing_dataset[int(index)]

    question = row[question_col]
    if choice_col is not None:
        if len(choice_col) == 1:
            options = [f"{chr(65 + i)}. {option}" for i, option in enumerate(row[choice_col[0]])]
        else:
            options = [f"{chr(65 + i)}. {row[option]}" for i, option in enumerate(choice_col)]

    if configs['generation']['k_self_consistency'] != False:
        k_paths = json.loads(row['response']) # unpack json and insert spaces for self consistency
        response = ""
        for k in range(configs['generation']['k_self_consistency']):
            response += f"Path {k}: \n\n {k_paths[k]} \n\n\n"
            response += "==================================="
    else: 
        response = row['response']
    return f"### \#{index+1}.\n\n### Question:\n{question}\n\n{options[0]}\n\n{options[1]}\n\n{options[2]}\n\n{options[3]}\n\n### Response:\n{response}"

def update_human_labelling(processed_answer, index, processed_answers, button):
    index = int(index)
    # Save the processed_answer to the dictionary
    if processed_answer != '': processed_answers[index] = processed_answer
    # Previous Empty: Find the previous unprocessed index
    if button == "Previous Empty":
        prev_index = find_unprocessed_index(processing_dataset, processed_answers, index - 1, prev=True)
        if prev_index < 0:
            # If beginning of dataset, inform the user
            display_text = "You have reached the beginning of the dataset."
            return display_text, gr.update(visible=False), prev_index, processed_answers
        else:
            # Display the previous question and response
            display_text = get_row(processing_dataset, prev_index)
            return display_text, gr.update(visible=True, value=""), prev_index, processed_answers
        
    # Previous: Find the previous index
    if button == "Previous":
        prev_index = index - 1
        if prev_index < 0:
            # If beginning of dataset, inform the user
            display_text = "You have reached the beginning of the dataset."
            return display_text, gr.update(visible=False), index, processed_answers
        else:
            display_text = get_row(processing_dataset, prev_index)
            return display_text, gr.update(visible=True, value=""), prev_index, processed_answers

    # Next: Find the next index
    if button == "Next":
        next_index = index + 1
        if next_index >= len(processing_dataset):
            # If end of dataset, inform the user
            display_text = "You have reached the end of the dataset."
            return display_text, gr.update(visible=False), next_index, processed_answers
        else:
            # Display the next question and response
            display_text = get_row(processing_dataset, next_index)
            return display_text, gr.update(visible=True, value=""), next_index, processed_answers
        
    # Next Empty: Find the next unprocessed index
    if button == "Next Empty":
        next_index = find_unprocessed_index(processing_dataset, processed_answers, index + 1)
        if next_index >= len(processing_dataset):
            # If end of dataset, inform the user
            display_text = "You have reached the end of the dataset."
            return display_text, gr.update(visible=False), next_index, processed_answers
        else:
            # Display the next question and response
            display_text = get_row(processing_dataset, next_index)
            return display_text, gr.update(visible=True, value=""), next_index, processed_answers

def save_dataset(processing_dataset, save_dir_input, processed_answers):
    # Function to update the dataset using map
    def add_processed_answer(example, idx):
        if idx in processed_answers:
            example['processed_answer'] = processed_answers[idx]
        return example

    # Update the dataset
    updated_dataset = processing_dataset.map(add_processed_answer, with_indices=True)
    # Save the updated dataset to a temporary directory first
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_save_path = os.path.join(tmpdirname, "dataset")
        updated_dataset.save_to_disk(temp_save_path)
        # Remove the old save_dir and replace it with the new one
        if os.path.exists(save_dir_input):
            shutil.rmtree(save_dir_input)
        shutil.move(temp_save_path, save_dir_input)
    return "Dataset saved successfully."

# run inference.py
def run_inference(*args):
    input_args = []
    tags = ['-m', '-d', '-f', '-t', '-s', '-p', '-r', '-b']
    for i in range(len(tags)):
        arg = ""
        arg += tags[i] 
        if i == 1:
            for j in args[i]:
                arg += j + " "
        else:
            arg += str(args[i])
        input_args.append(arg)

    try:
        command = [sys.executable, "inference.py"] + input_args
        os.makedirs("temp", exist_ok=True)
        with open("temp/inference.log", "w") as log_file:
            subprocess.run(command, stdout=log_file)
        return gr.update()

    except Exception as e:
        return gr.update(value=str(e))

def run_process_response(*args):
    os.environ['GROQ_API_KEY'] = args[0]
    input_args = [f"-d{args[1]}"]
    if args[2] != '': input_args.append(f"-s{args[2]}") 

    try: 
        command = [sys.executable, "process_response.py"] + input_args

        os.makedirs("temp", exist_ok=True)
        with open("temp/process_response.log", "w") as log_file:
            subprocess.run(command, stdout=log_file)

    except Exception as e:
        return gr.update(value=str(e))

def run_auto_eval(args):
    if args[3]: tag = 'SC'
    elif args[2]: tag = 'CoT'
    else: tag = str(args[1]) + '-shot'
    input_args = [f"-e{os.path.abspath('shortened/' + args[0] + '/' + tag)}"]
    try:
        command = [sys.executable, "eval.py"] + input_args
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            display_text = ''
            res = result.stdout.split(' ')
            for text in res:
                skipline = text.find('\n')
                if skipline != -1:
                    display_text += text[:skipline] + '\n\n'
                    display_text += text[skipline+1:] + ' '
                else:
                    display_text += text
            return [gr.update(visible=False), gr.update(visible=True, value=display_text), gr.update(visible=True), gr.update(), None, None]

        else:
            return [gr.update(visible=False), gr.update(visible=True, value=result.stderr), gr.update(visible=True), gr.update(), None, None]
        
    except Exception as e:
        return [gr.update(visible=False), gr.update(visible=True, value=str(e)), gr.update(visible=True), gr.update(), None, None]

def get_model_reponse(eval_models, cur_idx):
    display_text = ''
    for idx,  model in enumerate(eval_models):
        if model == 'length': continue # ignore the key storing the length of datasets 
        row = eval_models[model][cur_idx]

        if choice_col is not None:
            if len(choice_col) == 1:
                options = [f"{chr(65 + i)}. {option}" for i, option in enumerate(row[choice_col[0]])]
            else:
                options = [f"{chr(65 + i)}. {row[option]}" for i, option in enumerate(choice_col)]

        if idx == 0: 
            display_text += f'## Question {cur_idx+1}:\n{row[question_col]}\n\n'
            if choice_col is not None: display_text += f'{options[0]}\n\n{options[1]}\n\n{options[2]}\n\n{options[3]}\n'

        display_text += f'### {model}:\n'
        display_text += f"{row['response']}"

        if not row['processed_answer'].isalpha() or not (row['processed_answer'].lower() in choices or row['processed_answer'].upper() in choices): # unrelated responses 
            display_text += '❔\n'
            return display_text

        if choices == ['A', 'B', 'C', 'D']: # multiple choice 
            ans = ord(row['processed_answer']) - 65
        else: # decision 
            ans = row['processed_answer'].lower()
        
        if ans != row[answer_col]:
            display_text += '❌\n'
        else:
            display_text += '✅\n'

    display_text += f"\nCorrect answer: {chr(row[answer_col]+65) if choices == ['A', 'B', 'C', 'D'] else row[answer_col]}"

    return display_text

def update_human_eval(eval_models, cur_idx, button):
    if button == 'Previous': 
        cur_idx = max(0, cur_idx - 1)
    else:
        cur_idx = min(eval_models['length'], cur_idx + 1)
    display_text = get_model_reponse(eval_models, cur_idx)
    return [display_text, cur_idx]

def run_eval(*args):
    if args[0] == 'Auto': # auto evaluation 
        return run_auto_eval(args[2:])
    # human evaluation 
    if args[5]: tag = 'SC'
    elif args[4]: tag = 'CoT'
    else: tag = str(args[3]) + '-shot'

    # get the name of the columns of datasets 
    get_ds_info(args[1])

    path = tag + '/' + args[1]
    eval_models = {} # model_name: shortened.json, length: length of datasets 
    for model in models:
        if os.path.exists(os.path.abspath('shortened/' + model + '/' + path + '.json')):
            with open(os.path.abspath('shortened/' + model + '/' + path + '.json')) as f:
                eval_models[model] = json.load(f)
                eval_models['length'] = len(eval_models[model])

    display_text = get_model_reponse(eval_models, 0)
    return [gr.update(visible=False), gr.update(visible=True, value=display_text), gr.update(visible=True), gr.update(visible=True), 0, eval_models]

def update_prompting(selected_options):
    if "Few-shots prompting" in selected_options:
        return [gr.update(interactive=True), False, gr.update(interactive=False)]
    elif "Self Consistency" in selected_options:
        return [gr.update(interactive=False), False, gr.update(interactive=True)]
    return [gr.update(interactive=False), True, gr.update(interactive=False)]

def update_eval(selected_options):
    if "Auto" in selected_options:
        return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)]
    if "Human" in selected_options:
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
    
# Define Gradio interface
def create_ui():       
    with gr.Blocks(css=custom_css) as app:
        gr.Markdown("<h1 style='text-align: center;'>Medical Benchmarking Framework</h1>")
        with gr.Tabs():
            # Section 1: Inference
            with gr.Tab('Inference'):
                def update_inference_content():
                        with open("temp/process_response.log", 'r') as f:
                            content = f.read()
                        return content
                
                CoT = gr.State()
                with gr.Column() as inference_input_group:
                    model = gr.Radio(choices=[model for model in models], label="Select a model")
                    
                    dataset = gr.CheckboxGroup(choices=[dataset for dataset in datasets], label="Select datasets")
                    
                    prompting = gr.Radio(choices=[prompting for prompting in prompting_techniques], label="Prompting techniques")
                    fs = gr.Dropdown(choices=[0, 1, 2, 3, 4, 5], label="Few-shots prompting", value=0, interactive=False)
                    k_self_con = gr.Dropdown(choices=[0, 1, 2, 3, 4, 5], label="Self Consistency", value=0, interactive=False)
                    prompting.change(update_prompting, inputs=prompting, outputs=[fs, CoT, k_self_con])
                    prompting.change(update_prompting, inputs=prompting, outputs=[fs, CoT, k_self_con])
                    
                    gr.Markdown("### Generation parameters")
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.1, label="top_p", value=0.9)
                    temp = gr.Slider(minimum=0, maximum=2, step=0.1, label="temperature", value=1)
                    batch_size = gr.Dropdown(choices=[1, 2, 3, 4, 5], label="Batch size", value=3)

                    run_button_inference = gr.Button("Run inference")

                with gr.Column(visible=False) as inference_output_group:
                    return_button = gr.Button("Return", elem_id='return_button')
                    output_area_inference = gr.Textbox(label="Inference Output", interactive=False, lines=10)
                    update_inference = gr.Button("Update output")
                    
                    run_button_inference.click(lambda :[gr.update(visible=False), gr.update(visible=True)], 
                                               outputs=[inference_input_group, inference_output_group])
                    run_button_inference.click(run_inference, inputs=[model, dataset, fs, CoT, k_self_con, top_p, temp, batch_size], 
                                               outputs=[output_area_inference])
                    update_inference.click(update_inference_content, outputs=output_area_inference)
                    return_button.click(lambda :[gr.update(visible=True), gr.update(visible=False)], 
                                               outputs=[inference_input_group, inference_output_group])

            # Section 2: Response Processing
            with gr.Tab("Response Processing"):
                def update_process_content_auto():
                        with open("process_response.log", 'r') as f:
                            content = f.read()
                        return content
                
                def start_labeling(load_dir, save_dir):
                    if load_dir == '':
                        error_message = "Error: Load Directory cannot be empty."
                        return [gr.update(visible=True), gr.update(visible=False), gr.update(), None, None]
                    if save_dir == '':
                        error_message = "Error: Save Directory cannot be empty."
                        return [gr.update(visible=True), gr.update(visible=False), gr.update(), None, None]
                    if os.path.abspath(load_dir) == os.path.abspath(save_dir):
                        error_message = "Error: Load Directory and Save Directory must be different."
                        return [gr.update(visible=True), gr.update(visible=False), gr.update(), None, None]
                    
                    try:
                        processed_answers, start_index = load_dataset(load_dir, save_dir)
                    except Exception as e:
                        return [gr.update(visible=True), gr.update(visible=False), error_message, None, None]
                    
                    display_text = get_row(processing_dataset, start_index)
                    return [gr.update(visible=False), gr.update(visible=True), display_text, start_index, processed_answers]
                
                def save_progress(save_dir, processed_answers):
                    #print(processed_answers)
                    message = save_dataset(processing_dataset, save_dir, processed_answers)
                    return message
                
                index_state = gr.State()
                processed_answers_state = gr.State()
                with gr.Column() as process_input_group:
                    groq_key = gr.Textbox(label="Groq API key", placeholder="Enter GROQ API key here for automatic processing")
                    load_dir = gr.Textbox(label="Load Directory", placeholder="Enter the path to the load directory")
                    save_dir = gr.Textbox(label="Save Directory", placeholder="Enter the path to the save directory")
                    with gr.Row():
                        auto_process_button = gr.Button("Auto Process", scale=0)
                        human_labelling_button = gr.Button("Human Labelling", scale=0)  

                with gr.Column(visible=False) as process_output_group:
                    return_button = gr.Button("Return", elem_id='return_button', visible=False)
                    output_area_process = gr.Textbox(label="Response Process Output", interactive=False, lines=10)
                    update_process_button = gr.Button("Update output")                

                    auto_process_button.click(lambda :[gr.update(visible=False), gr.update(visible=True)], 
                                              outputs=[process_input_group, process_output_group])
                    auto_process_button.click(run_process_response, inputs=[groq_key, load_dir, save_dir], 
                                              outputs=[output_area_process])
                    update_process_button.click(update_process_content_auto, outputs=output_area_process)

                with gr.Column(visible=False) as human_process_output_group:
                    return_button = gr.Button("Return", elem_id='return_button')
                    display = gr.Markdown()
                    processed_answer_input = gr.Textbox(label="Processed Answer")

                    with gr.Row():
                        prevEmpty_button = gr.Button("Previous Empty", elem_id="prevEmpty_button", scale=0)
                        prev_button = gr.Button("Previous", elem_id="prev_button", scale=0)  
                        save_button = gr.Button("Save Dataset", elem_id="save_button", scale=0)
                        next_button = gr.Button("Next", elem_id="next_button", scale=0)
                        nextEmpty_button = gr.Button("Next Empty", elem_id="nextEmpty_button", scale=0)

                    save_output = gr.Textbox(label="Save Status", interactive=False, visible=False)

                return_button.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], 
                                    outputs=[process_input_group, process_output_group, human_process_output_group])

                human_labelling_button.click(
                    start_labeling,
                    inputs=[load_dir, save_dir],
                    outputs=[process_input_group, human_process_output_group, display, index_state, processed_answers_state]
                )
                prevEmpty_button.click(
                    update_human_labelling,
                    inputs=[processed_answer_input, index_state, processed_answers_state, prevEmpty_button],
                    outputs=[display, processed_answer_input, index_state, processed_answers_state]
                )
                prev_button.click(
                    update_human_labelling,
                    inputs=[processed_answer_input, index_state, processed_answers_state, prev_button],
                    outputs=[display, processed_answer_input, index_state, processed_answers_state]
                )
                next_button.click(
                    update_human_labelling,
                    inputs=[processed_answer_input, index_state, processed_answers_state, next_button],
                    outputs=[display, processed_answer_input, index_state, processed_answers_state]
                )
                nextEmpty_button.click(
                    update_human_labelling,
                    inputs=[processed_answer_input, index_state, processed_answers_state, nextEmpty_button],
                    outputs=[display, processed_answer_input, index_state, processed_answers_state]
                )

                save_button.click(
                    fn=save_progress,
                    inputs=[save_dir, processed_answers_state],
                    outputs=save_output
                )


            # Section 3: Evaluation
            with gr.Tab("Evaluation"):
                index_eval = gr.State()
                eval_models = gr.State()

                with gr.Column() as eval_input_group:
                    eval_mode = gr.Radio(choices=['Auto', 'Human'], label='Evaluation mode')

                    model = gr.Radio(choices=[model for model in models], label="Select a model", visible=False)
                    dataset = gr.Radio(choices=[dataset for dataset in datasets], label="Select a dataset", visible=False, interactive=True)
                    with gr.Group(visible=False) as prompt_group:
                        prompting = gr.Radio(choices=[prompting for prompting in prompting_techniques], label="Prompting techniques")
                        fs = gr.Dropdown(choices=[0, 1, 2, 3, 4, 5], label="Few-shots prompting", value=0, interactive=False)
                        k_self_con = gr.Dropdown(choices=[0, 1, 2, 3, 4, 5], label="Self Consistency", value=0, interactive=False)
                        run_button_eval = gr.Button("Run evaluation")
                        prompting.change(update_prompting, inputs=prompting, outputs=[fs, CoT, k_self_con])
                        prompting.change(update_prompting, inputs=prompting, outputs=[fs, CoT, k_self_con])

                eval_mode.change(update_eval, inputs=eval_mode, outputs=[model, dataset, prompt_group])
                
                return_button = gr.Button("Return", elem_id='return_button', visible=False)
                output_area_eval = gr.Markdown(visible=False)
                
                with gr.Row(visible=False) as nav_group:

                    prev_button = gr.Button("Previous", elem_id="prev_button", scale=0) 
                    next_button = gr.Button("Next", elem_id="next_button", scale=0)

                run_button_eval.click(run_eval, inputs=[eval_mode, dataset, model, fs, CoT, k_self_con], 
                                        outputs=[eval_input_group, output_area_eval, return_button, nav_group, index_eval, eval_models])
                prev_button.click(update_human_eval, inputs=[eval_models, index_eval, prev_button], outputs=[output_area_eval, index_eval])
                next_button.click(update_human_eval, inputs=[eval_models, index_eval, next_button], outputs=[output_area_eval, index_eval])
                return_button.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)], 
                                    outputs=[eval_input_group, output_area_eval, return_button, nav_group])

    app.launch()


# Run the UI
if __name__ == "__main__":
    create_ui()