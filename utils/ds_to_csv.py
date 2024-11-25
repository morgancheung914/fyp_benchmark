from datasets import load_from_disk
import argparse
# Load the dataset from disk

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Input dataset file path (hf datasets format)')
parser.add_argument('-o', '--output', type=str, help='Output csv file path')
args = parser.parse_args()
input_path = args.input
output_path = args.output

dataset = load_from_disk(input_path)

# Save the dataset as a CSV file
dataset.to_csv(output_path)
