import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import get_logger

logger = get_logger("Explainer")

class Explainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.explain_dir = "results/explainability"
        os.makedirs(self.explain_dir, exist_ok=True)
        
    def explain_url(self, url, filename="saliency.png"):
        """Simple input-gradient saliency explanation for a URL."""
        encoded = self.tokenizer.encode(url)
        inputs = torch.tensor([encoded], dtype=torch.long, device=self.device)
        
        # We need gradients with respect to the embeddings
        embeddings = self.model.embedding(inputs) # (1, max_len, embed_dim)
        embeddings.retain_grad()
        
        # Forward pass manually through the rest of the CNN model
        # Assuming model is CNN1DPhishDetector
        x = embeddings.transpose(1, 2)
        x1 = torch.nn.functional.relu(self.model.conv1(x))
        x2 = torch.nn.functional.relu(self.model.conv2(x))
        x3 = torch.nn.functional.relu(self.model.conv3(x))
        
        x1 = self.model.pool(x1).squeeze(-1)
        x2 = self.model.pool(x2).squeeze(-1)
        x3 = self.model.pool(x3).squeeze(-1)
        
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_drop = self.model.dropout(x_cat)
        x_fc1 = torch.nn.functional.relu(self.model.fc1(x_drop))
        logits = self.model.fc2(x_fc1)
        
        # Backward pass
        self.model.zero_grad()
        logits.backward()
        
        # Get gradient magnitudes
        grads = embeddings.grad[0].cpu().numpy() # (max_len, embed_dim)
        saliency = np.linalg.norm(grads, axis=1) # (max_len,)
        
        # Normalize
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
            
        # Truncate visualization to actual URL length
        url_len = min(len(url), self.tokenizer.max_len)
        saliency = saliency[:url_len]
        chars = list(url[:url_len])
        
        self.plot_saliency(chars, saliency, filename)
        return chars, saliency
        
    def plot_saliency(self, chars, saliency, filename):
        plt.figure(figsize=(min(20, len(chars) * 0.3), 3))
        # Ensure 2D array for heatmap
        sns.heatmap([saliency], annot=[chars], fmt="", cmap="Reds", 
                    cbar=False, xticklabels=False, yticklabels=False)
        plt.title("Character Importance (Input-Gradients)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.explain_dir, filename), bbox_inches='tight')
        plt.close()
