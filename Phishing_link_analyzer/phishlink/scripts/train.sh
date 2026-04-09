#!/bin/bash
# scripts/train.sh

# Exit on error
set -e

# Change directory to the root of the project
cd "$(dirname "$0")/.."

LOG_FILE="logs/train.log"
mkdir -p logs

echo "Starting training job..." | tee -a $LOG_FILE

# Run the trainer script
python -m src.trainer >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "Training completed successfully." | tee -a $LOG_FILE
else
    echo "Training failed. Check logs/train.log." | tee -a $LOG_FILE
fi
