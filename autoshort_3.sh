#!/bin/bash

# Check if groq_j.txt exists
if [ ! -f groq_j.txt ]; then
    echo "groq_j.txt not found. Please create the file with API keys, each on a new line."
    exit 1
fi

# Loop through each line (API key) in groq_j.txt only once
while IFS= read -r api_key || [ -n "$api_key" ]; do
    # Export the API key as an environment variable
    export GROQ_API_KEY="$api_key"
    echo "Using GROQ_API_KEY: $GROQ_API_KEY"

    # Run the Python script with the specified config and log output
    log_file="output_$(date +'%Y%m%d')_llama.log"
    python process_response.py -c config.yaml >> "$log_file" 2>&1

    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error running process_response.py with GROQ_API_KEY: $GROQ_API_KEY. Check $log_file for details."
    fi

    # Sleep for a short time to avoid spamming requests, if necessary
    sleep 1
done < groq_j.txt

echo "Finished processing all API keys in groq_j.txt."
