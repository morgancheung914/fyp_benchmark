import gradio as gr
from datasets import load_from_disk, Dataset
import os
import shutil
import tempfile

def load_dataset(load_dir, save_dir):
    # Always load the original dataset from load_dir
    dataset = load_from_disk(load_dir)
    print(f"Loaded original dataset from: {load_dir}")

    # Add 'index' column if it doesn't exist
    if 'index' not in dataset.column_names:
        dataset = dataset.add_column('index', list(range(len(dataset))))

    # Initialize 'processed_answer' column if it doesn't exist
    if 'processed_answer' not in dataset.column_names:
        dataset = dataset.add_column('processed_answer', [''] * len(dataset))

    # Initialize processed_answers dictionary
    processed_answers = {}

    if os.path.exists(save_dir) and os.listdir(save_dir):
        # Load the saved dataset to get existing processed answers
        saved_dataset = load_from_disk(save_dir)
        print(f"Loaded saved dataset from: {save_dir}")

        # Create a mapping from index to processed_answer
        index_to_processed = {
            example['index']: example['processed_answer']
            for example in saved_dataset
            if example['processed_answer']
        }

        # Update the processed_answers dictionary
        for idx, example in enumerate(dataset):
            entry_index = example['index']
            if entry_index in index_to_processed:
                processed_answers[idx] = index_to_processed[entry_index]

        # Find the index of the last processed entry
        if processed_answers:
            last_processed_idx = max(processed_answers.keys())
            start_index = last_processed_idx + 1  # Start from the next entry
        else:
            start_index = 0
    else:
        print("No existing saved dataset found. Starting fresh.")
        start_index = 0

    return dataset, processed_answers, start_index

def find_next_unprocessed_index(dataset, processed_answers, start=0):
    # Find the index of the next entry without a 'processed_answer'
    for i in range(start, len(dataset)):
        if i not in processed_answers:
            return i
    return len(dataset)  # Return length if all entries are processed

def get_row(dataset, index):
    if index >= len(dataset):
        return "You have reached the end of the dataset."
    # Retrieve index, question, and response for the current index
    row = dataset[int(index)]
    index_value = row.get('index', 'N/A')
    question = row['question']
    response = row['response']
    return f"### Index: {index_value}\n\n### Question:\n{question}\n\n### Response:\n{response}"

def update(processed_answer, index, processed_answers):
    index = int(index)
    # Save the processed_answer to the dictionary
    processed_answers[index] = processed_answer
    # Find the next unprocessed index
    next_index = find_next_unprocessed_index(dataset, processed_answers, index + 1)
    if next_index >= len(dataset):
        # If end of dataset, inform the user
        display_text = "You have reached the end of the dataset."
        return display_text, gr.update(visible=False), next_index, processed_answers
    else:
        # Display the next question and response
        display_text = get_row(dataset, next_index)
        return display_text, gr.update(value=""), next_index, processed_answers

def save_dataset(dataset, save_dir_input, processed_answers):
    # Function to update the dataset using map
    def add_processed_answer(example, idx):
        if idx in processed_answers:
            example['processed_answer'] = processed_answers[idx]
        return example

    # Update the dataset
    updated_dataset = dataset.map(add_processed_answer, with_indices=True)
    # Save the updated dataset to a temporary directory first
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_save_path = os.path.join(tmpdirname, "dataset")
        updated_dataset.save_to_disk(temp_save_path)
        # Remove the old save_dir and replace it with the new one
        if os.path.exists(save_dir_input):
            shutil.rmtree(save_dir_input)
        shutil.move(temp_save_path, save_dir_input)
    return "Dataset saved successfully."

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Dataset Loader")
        load_dir_input = gr.Textbox(label="Load Directory", placeholder="Enter the path to the load directory")
        save_dir_input = gr.Textbox(label="Save Directory", placeholder="Enter the path to the save directory")
        start_button = gr.Button("Start Labeling")

        # Components for the labeling interface
        display = gr.Markdown(visible=False)
        processed_answer_input = gr.Textbox(label="Processed Answer", visible=False)
        next_button = gr.Button("Submit and Next", visible=False)
        save_button = gr.Button("Save Dataset", visible=False)
        save_output = gr.Textbox(label="Save Status", interactive=False, visible=False)

        index_state = gr.State()
        processed_answers_state = gr.State()

        def start_labeling(load_dir, save_dir):
            # Check if load_dir and save_dir are the same
            if os.path.abspath(load_dir) == os.path.abspath(save_dir):
                error_message = "Error: Load Directory and Save Directory must be different."
                return [gr.update(value=error_message), gr.update(), gr.update(),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        None, None]
            global dataset  # Declare dataset as global
            dataset, processed_answers, start_index = load_dataset(load_dir, save_dir)
            index = start_index
            if index >= len(dataset):
                display_text = "All entries have been processed."
                return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True, value=display_text),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=True),
                        index, processed_answers]
            else:
                display_text = get_row(dataset, index)
                return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True, value=display_text),
                        gr.update(visible=True), gr.update(visible=True),
                        gr.update(visible=True), gr.update(visible=True),
                        index, processed_answers]

        start_button.click(
            fn=start_labeling,
            inputs=[load_dir_input, save_dir_input],
            outputs=[load_dir_input, save_dir_input, start_button,
                     display, processed_answer_input, next_button, save_button, save_output,
                     index_state, processed_answers_state]
        )

        def next_step(processed_answer, index, processed_answers):
            return update(processed_answer, index, processed_answers)

        next_button.click(
            fn=next_step,
            inputs=[processed_answer_input, index_state, processed_answers_state],
            outputs=[display, processed_answer_input, index_state, processed_answers_state]
        )

        def save_progress(save_dir, processed_answers):
            message = save_dataset(dataset, save_dir, processed_answers)
            return message

        save_button.click(
            fn=save_progress,
            inputs=[save_dir_input, processed_answers_state],
            outputs=save_output
        )

    demo.launch()

if __name__ == "__main__":
    main()
