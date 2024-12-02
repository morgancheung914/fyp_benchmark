import gradio as gr
import subprocess
import sys
import os

# Function to run a script with user-provided arguments (generic for Python or Shell scripts)
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

        # Execute the script
        with open("inference.log", "w") as log_file:
            subprocess.run(command, stdout=log_file)
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

def run_process_response(*args):
    print(args)
    os.environ['GROQ_API_KEY'] = args[0]
    input_args = [f"-d{args[1]}"]
    if args[2] != '': input_args.append(f"-s{args[2]}") 

    try: # process_response script 
        command = [sys.executable, "process_response.py"] + input_args
        print(command)
        with open("process_response.log", "w") as log_file:
            subprocess.run(command, stdout=log_file)

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def run_eval(*args):
    input_args = [f"-e{args[0]}"]
    try:
        command = [sys.executable, "eval.py"] + input_args
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            return f"{result.stdout}"

        else:
            return f"Error: {result.stderr}"
        
    except Exception as e:
        return f"Error occurred: {str(e)}"


# Define Gradio interface
def create_ui():       
    models = ['ChatGLM', 'Internist', 'Llama3', 'Med42', 'Meditron']
    datasets = ['MMLU_biology', 'MMLU_anatomy', 'MMLU_medicine', 'MMLU_clinical', 'PubMedQA', 'MedMCQA', 'HaluEval']

    with gr.Blocks() as app:
        gr.Markdown("<h1 style='text-align: center;'>Medical Benchmarking Framework</h1>")
        
        # Section 1: Inference
        with gr.Column():
            gr.Markdown("# Inference")
            model = gr.Radio(choices=[model for model in models], label="Select a model")
            
            dataset = gr.CheckboxGroup(choices=[dataset for dataset in datasets], label="Select datasets")
            
            gr.Row(dataset)

            fs = gr.Dropdown(choices=[0, 1, 2, 3, 4, 5], label="Few-shots prompting", value=0)

            CoT = gr.Checkbox(label="Chain of Thoughts")

            k_self_con = gr.Dropdown(choices=[0, 1, 2, 3, 4, 5], label="Self Consistency", value=0)

            gr.Markdown("### Generation parameters")
            top_p = gr.Slider(minimum=0, maximum=1, step=0.1, label="top_p", value=0.9)
            temp = gr.Slider(minimum=0, maximum=2, step=0.1, label="temperature", value=1)
            batch_size = gr.Dropdown(choices=[0, 1, 2, 3, 4, 5], label="Batch size", value=3)

            run_button_inference = gr.Button("Run inference")
            output_area_inference = gr.Textbox(label="Inference Output", interactive=False, lines=10)
            update_inference = gr.Button("Update output")

            def update_inference_content():
                with open("inference.log", 'r') as f:
                    content = f.read()
                return content
            
            run_button_inference.click(run_inference, inputs=[model, dataset, fs, CoT, k_self_con, top_p, temp, batch_size])
            update_inference.click(update_inference_content, outputs=output_area_inference)
            

        # Section 2: Run Script 2
        with gr.Column():
            gr.Markdown("# Response processing")
            groq_key = gr.Textbox(label="Groq API key", placeholder="Enter GROQ API key here")

            load_dir = gr.Textbox(label="Load Directory", placeholder="Enter the path to the load directory")
            save_dir = gr.Textbox(label="Save Directory", placeholder="Enter the path to the save directory")
            run_button_process = gr.Button("Run Response Process")
            output_area_process = gr.Textbox(label="Response Process Output", interactive=False, lines=10)
            update_process = gr.Button("Update output")

            def update_process_content():
                with open("process_response.log", 'r') as f:
                    content = f.read()
                return content
    
            run_button_process.click(run_process_response, inputs=[groq_key, load_dir, save_dir])
            update_process.click(update_process_content, outputs=output_area_process)

           
        # Section 3: Evaluation
        with gr.Column():
            gr.Markdown("# Evaluation")
            
            eval_path = gr.Textbox(label="Evaluating path", placeholder="Enter path to folder for model responses")
            run_button_3 = gr.Button("Run evaluation")
            output_area_eval = gr.Textbox(label="Evaluation Output", interactive=False, lines=10)

            run_button_3.click(run_eval, inputs=[eval_path], outputs=output_area_eval)

    app.launch()


# Run the UI
if __name__ == "__main__":
    create_ui()