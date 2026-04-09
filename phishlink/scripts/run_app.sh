#!/bin/bash
# scripts/run_app.sh

# Exit on error
set -e

# Change directory to the root of the project
cd "$(dirname "$0")/.."

LOG_FILE="logs/app.log"
mkdir -p logs

echo "Starting Streamlit App..." | tee -a $LOG_FILE
streamlit run app/streamlit_app.py --server.port 8501 >> $LOG_FILE 2>&1
