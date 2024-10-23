#!/bin/bash

# Constants for the fixed arguments
MODEL="Med42"  # Replace with your fixed model argument for -m
N_ARGUMENT="MedMCQA"  # Replace with your fixed argument for -n

# Log file for the script's output
LOG_FILE="as2.log"

# Function to run the Python script and retry if it fails
run_python_script() {
    PART=$1
    RETRY_COUNT=0
    MAX_RETRIES=3

    # Command to run the Python script with the current part
    CMD="python3 process_response.py -d responses/Med42/MedMCQA-partitioned/$PART -m $MODEL -n $N_ARGUMENT -s shortened/ChatGLM_MedMCQA_partition/$PART"

    while [ $RETRY_COUNT -le $MAX_RETRIES ]; do
        echo "Running: $CMD" | tee -a "$LOG_FILE"
        source /Users/morgan/opt/miniconda3/etc/profile.d/conda.sh
        conda activate fypb
        # Run the command and capture the exit code
        $CMD >> "$LOG_FILE" 2>&1
        EXIT_CODE=$?

        # Check if the script ran successfully
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Script succeeded for part: $PART" | tee -a "$LOG_FILE"
            break
        else
            echo "Script failed for part: $PART with exit code $EXIT_CODE. Retrying... (Attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES)" | tee -a "$LOG_FILE"
            ((RETRY_COUNT++))
            sleep 2  # Optional: wait for 2 seconds before retrying
        fi

        # If max retries reached, log and stop retrying
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo "Script failed for part: $PART after $MAX_RETRIES attempts. Moving on to next part." | tee -a "$LOG_FILE"
            break
        fi
    done
}

# Loop through parts 0 to 10
for i in {0..10}; do
    PART="part_$i"
    run_python_script "$PART"
done