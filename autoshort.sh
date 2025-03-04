#!/bin/bash
# Check if groq.txt exists
if [ ! -f groq.txt ]; then
    echo "groq.txt not found. Please create the file with API keys, each on a new line."
    exit 1
fi

# Convert the line endings of groq.txt to Unix format (in case it has CRLF)
dos2unix groq.txt

# Get the current date for the log file
current_date=$(date +'%Y%m%d')
log_file="output_${current_date}.log"

# Loop through each line (API key) in groq.txt only once
while IFS= read -r api_key || [[ -n "$api_key" ]]; do
    # Skip empty lines
    if [[ -z "$api_key" ]]; then
        continue
    fi

    # Export the API key as an environment variable
    export GROQ_API_KEY="$api_key"
    echo "Using GROQ_API_KEY: $GROQ_API_KEY"

    # Run the Python script with the specified config and log output
    python3 process_response.py -c config.yaml >> "$log_file" 2>&1

    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error running process_response.py with GROQ_API_KEY: $GROQ_API_KEY. Check $log_file for details."
        continue
    fi

    # Sleep for a short time to avoid spamming requests, if necessary
    sleep 1
done < groq.txt

echo "Finished processing all API keys in groq.txt."