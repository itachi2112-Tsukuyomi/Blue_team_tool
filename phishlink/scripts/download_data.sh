#!/bin/bash
# scripts/download_data.sh

# Exit on error
set -e

# Change directory to the root of the project
cd "$(dirname "$0")/.."

LOG_FILE="logs/data_download.log"
mkdir -p logs data/raw data/processed

echo "Starting data acquisition..." | tee -a $LOG_FILE

# Method 1: Hugging Face (oaicite:0) will be handled in Python data_loader.py,
# but we document the fallback steps here or trigger them if needed.
# Since python handles HF and Kaggle via API, this script mainly orchestrates
# running the data_loader script and logging.

echo "Running python data pipeline to fetch and process datasets..." | tee -a $LOG_FILE

# Run the data_loader module directly to execute the pipeline.
python -m src.data_loader >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "Data pipeline completed successfully." | tee -a $LOG_FILE
else
    echo "Data pipeline failed or fell back to synthetic data. Check logs/data_download.log." | tee -a $LOG_FILE
fi
