from datasets import load_from_disk
import os

# Load your dataset (replace 'your_dataset_name' with the actual dataset)
dataset = load_from_disk('responses/Med42/MedMCQA')

# Create a directory to save the partitions if it doesn't exist
output_dir = 'responses/Med42/MedMCQA-partitioned'
os.makedirs(output_dir, exist_ok=True)

# Split the dataset into 8 parts using shard and save each part
for i in range(20):
    part = dataset.shard(num_shards=20, index=i)
    
    # Save each partition to a directory (in JSON format or any format of your choice)
    part.save_to_disk(os.path.join(output_dir, f'part_{i}'))

print("Dataset has been partitioned and saved successfully.")
