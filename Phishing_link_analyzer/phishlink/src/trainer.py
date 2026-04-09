import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import get_logger

logger = get_logger("Trainer", log_file="logs/train.log")

class URLDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.urls = df['url'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        encoded = self.tokenizer.encode(url)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-3))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.5)
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
    def train(self, train_loader, val_loader, save_path="models/url_detector.pth"):
        best_val_loss = float('inf')
        patience_counter = 0
        epochs = self.config.get('epochs', 30)
        patience = self.config.get('patience', 5)
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(-1)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                pbar.set_postfix({"loss": loss.item()})
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved new best model with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break
                    
        # Save training history
        os.makedirs("results", exist_ok=True)
        with open("results/history.json", "w") as f:
            json.dump(self.history, f, indent=4)
        logger.info("Training history saved.")

    def evaluate(self, loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).squeeze(-1)
                
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= len(loader.dataset)
        val_acc = correct / total
        return val_loss, val_acc

def run_training():
    from .config import CONFIG
    from .utils import set_seed, save_metadata
    from .tokenizer import URLTokenizer
    from .model_builder import ModelBuilder
    
    set_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    
    # Check GPU and adjust batch size if necessary
    batch_size = CONFIG["batch_size"]
    if device.type == "cpu":
        batch_size = 32
        
    logger.info(f"Using device: {device}, Batch size: {batch_size}")
    
    # Load data
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    except Exception as e:
        logger.error("Could not load processed data. Please run data_loader first.")
        return
        
    # Initialize and fit tokenizer
    tokenizer = URLTokenizer(max_len=CONFIG["max_len"])
    tokenizer.fit_on_texts(train_df['url'].tolist())
    os.makedirs("models", exist_ok=True)
    tokenizer.save("models/tokenizer.json")
    
    # Create datasets and loaders
    train_dataset = URLDataset(train_df, tokenizer)
    val_dataset = URLDataset(val_df, tokenizer)
    test_dataset = URLDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build Model
    model = ModelBuilder.build_cnn(num_chars=tokenizer.vocab_size, 
                                   embedding_dim=CONFIG["embedding_dim"], 
                                   max_len=CONFIG["max_len"])
                                   
    # Train
    trainer = Trainer(model, CONFIG, device)
    trainer.train(train_loader, val_loader)
    
    # Evaluate & Calibrate
    from .evaluator import Evaluator
    model.load_state_dict(torch.load("models/url_detector.pth", map_location=device))
    evaluator = Evaluator(model, device, max_len=CONFIG["max_len"])
    evaluator.calibrate(val_loader)
    evaluator.evaluate(test_loader)
    
    # Save Metadata
    metadata = {
        "dataset_source": "Hugging Face or Synthetic",
        "seed": CONFIG["seed"],
        "hyperparams": CONFIG,
        "model_architecture": "CNN1DPhishDetector"
    }
    save_metadata("results/metadata.json", metadata)

if __name__ == "__main__":
    run_training()
