# Phishlink: Deep Learning URL Detector

A complete, reproducible, object-oriented Python project that implements a Phishing Link (URL) detector using Deep Learning (PyTorch) and exposes it via a Streamlit dashboard.

## 🚀 Features
- **Automated Data Acquisition**: Fetches data programmatically from Hugging Face `ealvaradob/phishing-dataset` (or falls back to built-in synthetic balancing generator if network/API blocked).
- **Deep Learning Options**: Configurable 1D-CNN or BiLSTM sequence modeling on Character-level URL tokens.
- **Explainability**: Integrated character-level saliency maps via input-gradients to highlight suspicious URL substrings.
- **Probability Calibration**: Uses Platt scaling on model logits to output true Confidence % probabilities, classifying into precise risk bands.
- **Streamlit Dashboard**: Easy-to-use GUI for single URL predictions or batch CSV uploads.
- **Rigorous Evaluation**: Built-in scripts to auto-generate ROC, PR Curve, Confusion matrix, and training history plots.

## 📁 Repository Structure
```
phishlink/
├── app/
│   └── streamlit_app.py       # Streamlit GUI for inference
├── data/
│   ├── processed/             # Cleaned and split CSVs (Train/Val/Test)
│   └── raw/                   # Raw downloaded datasets
├── logs/                      # Execution logs
├── models/                    # Saved weights, tokenizer, and calibrator
├── results/
│   ├── explainability/        # Generated Saliency Map visuals
│   ├── figures/               # ROC, Confusion Matrix, Calibration plots
│   ├── history.json           # Train loss and val accuracy history
│   ├── metadata.json          # Reproducibility constraints (seed, config)
│   └── metrics.csv            # Accuracy, Precision, F1, ROC-AUC
├── scripts/
│   ├── download_data.sh       # Script to trigger data download and cleaning
│   ├── run_app.sh             # Script to run Streamlit
│   └── train.sh               # Script to trigger the Trainer
├── src/
│   ├── config.py              # Hyperparameter config and constraints
│   ├── data_loader.py         # Data fetcher, cleaner, and splitter
│   ├── evaluator.py           # Evaluation, figures, and Platt Scaling
│   ├── explain.py             # Saliency map generator
│   ├── model_builder.py       # 1D-CNN and BiLSTM implementations
│   ├── tokenizer.py           # Character-level NLP Tokenizer
│   ├── trainer.py             # Main PyTorch Training loop
│   └── utils.py               # Utilities (Logger, seed setting)
├── tests/
│   ├── test_data_loader.py    # Unit tests for loading/processing
│   └── test_tokenizer.py      # Unit tests for NLP functions
├── README.md
└── requirements.txt
```

## 🛠️ Installation & Setup

1. **Clone the repo (or copy the directories)**
   ```bash
   cd phishlink
   ```

2. **Install dependencies**
   Ensure you have Python 3.8+ installed. 
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: PyTorch will install CPU version by default via standard pip; for GPU, setup your local CUDA toolkit matching your torch version).*

## 🏃‍♂️ Step-by-Step Reproduction

### 1. Data Downloading & Preprocessing
```bash
bash scripts/download_data.sh
```
OR
```bash
python -m src.data_loader
```
*Fallback logic: If Hugging Face is inaccessible due to firewall or lack of connection, the system will automatically trap the error and generate a functional Mock Synthetic Dataset to allow the whole pipeline to execute locally as scaffolding.*

### 2. Model Training
```bash
bash scripts/train.sh
```
OR
```bash
python -m src.trainer
```
- Fits the character tokenizer on the train set.
- Initializes the 1D-CNN character model.
- Trains using Early Stopping.
- Generates all evaluation metrics, probability calibration model, and explains predictions.
- **Output Artifacts generated:** `models/url_detector.pth`, `models/tokenizer.json`, `models/calibrator.pkl`, `results/figures/*.png`.

### 3. Running Unit Tests
Validate that the tokenizer and data processor behave correctly:
```bash
pytest tests/
```

### 4. Running the Dashboard
```bash
bash scripts/run_app.sh
```
OR
```bash
streamlit run app/streamlit_app.py
```
This loads up the locally hosted Streamlit URL (default `localhost:8501`). You can type a URL to instantly see the raw probability, calibrated risk percentage, and generated explainability map.

## 📝 Configuration Decisions
Settings can be tweaked inside `src/config.py`:
- `max_len = 200`: URLs longer than 200 characters are truncated. Most phishing features exist in substring combinations within the domain route.
- `embedding_dim = 64`: Enough capacity to represent character abstractions without massive overfitting.
- `batch_size = 128` (Fallback to `32` on CPU).
- `seed = 42`: Handled globally for PyTorch, NumPy, and randomness to ensure strict reproducibility.

## 📧 Setting Up Email Alerts
The Network Monitoring tab includes an alerting feature that can notify you by email when someone on the network accesses a phishing URL. Since Google disables standard password authentication for secure apps, you'll need to generate an **App Password**:

1. Log into the Google Account you wish to send emails from (`Sender Gmail Address`).
2. Go to **Manage your Google Account** -> **Security**.
3. Under "How you sign in to Google", ensure **2-Step Verification** is turned ON.
4. Go to **2-Step Verification** and scroll down to **App passwords**.
5. Create a new App password (e.g., named "Phishlink"). 
6. Copy the generated 16-character password.
7. In the Phishlink Streamlit dashboard under the "Network Monitor" tab, enter your Gmail address and paste this 16-character password into the **Sender App Password** field.
