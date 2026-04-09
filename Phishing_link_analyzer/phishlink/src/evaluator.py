import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

from .utils import get_logger

logger = get_logger("Evaluator", log_file="logs/train.log")

class Evaluator:
    def __init__(self, model, device, max_len=200):
        self.model = model
        self.device = device
        self.model.eval()
        self.max_len = max_len
        self.calibrator = None
        
        self.fig_dir = "results/figures"
        os.makedirs(self.fig_dir, exist_ok=True)
        
    def get_predictions(self, loader):
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                
        return np.array(all_labels), np.array(all_probs)
        
    def calibrate(self, val_loader, save_path="models/calibrator.pkl"):
        """Train a Platt scaling calibrator on validation set predictions."""
        logger.info("Training probability calibrator (Platt scaling)...")
        y_val, val_probs = self.get_predictions(val_loader)
        
        # Logistic Regression on model probabilities
        calibrator = LogisticRegression()
        # Reshape for sklearn
        X_val = np.clip(val_probs, 1e-7, 1 - 1e-7)
        # Convert to logits to train Platt scaling
        X_val_logits = np.log(X_val / (1 - X_val)).reshape(-1, 1)
        
        calibrator.fit(X_val_logits, y_val)
        self.calibrator = calibrator
        
        with open(save_path, "wb") as f:
            pickle.dump(self.calibrator, f)
        logger.info(f"Calibrator saved to {save_path}")
        
    def evaluate(self, test_loader):
        """Evaluate model on testing data and generate plots."""
        y_true, y_probs = self.get_predictions(test_loader)
        
        if self.calibrator:
            # Calibrate probabilities
            X_test_logits = np.log(np.clip(y_probs, 1e-7, 1 - 1e-7) / (1 - np.clip(y_probs, 1e-7, 1 - 1e-7))).reshape(-1, 1)
            y_probs_calibrated = self.calibrator.predict_proba(X_test_logits)[:, 1]
            y_probs = y_probs_calibrated
            
        y_pred = (y_probs > 0.5).astype(int)
        
        # Metrics
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_probs)
        }
        
        logger.info(f"Test Metrics: {results}")
        pd.DataFrame([results]).to_csv("results/metrics.csv", index=False)
        
        self.plot_roc_curve(y_true, y_probs)
        self.plot_pr_curve(y_true, y_probs)
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_calibration(y_true, y_probs)
        self.plot_training_curves()
        
        return results
        
    def plot_roc_curve(self, y_true, y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Model ROC Curve (AUC = {auc:.3f})', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess Baseline', color='darkorange')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.fig_dir, "roc_curve.png"), bbox_inches='tight')
        plt.close()
        
    def plot_pr_curve(self, y_true, y_probs):
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        # Calculate baseline (ratio of positive class)
        baseline = sum(y_true) / len(y_true) if len(y_true) > 0 else 0.5
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Model PR Curve', color='blue')
        plt.axhline(y=baseline, color='darkorange', linestyle='--', label=f'Random Guess Baseline ({baseline:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.fig_dir, "precision_recall.png"), bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Phishing'], yticklabels=['Benign', 'Phishing'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.fig_dir, "conf_matrix.png"), bbox_inches='tight')
        plt.close()
        
    def plot_calibration(self, y_true, y_probs):
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Plot (Reliability Diagram)')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "calibration_plot.png"), bbox_inches='tight')
        plt.close()
        
    def plot_training_curves(self):
        try:
            with open("results/history.json", "r") as f:
                history = json.load(f)
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['val_acc'], label='Val Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Curve')
            plt.legend()
            
            plt.savefig(os.path.join(self.fig_dir, "training_curve.png"), bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot training curves: {e}")
