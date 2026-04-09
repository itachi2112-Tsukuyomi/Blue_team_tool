import json
from collections import Counter
import numpy as np

class URLTokenizer:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.oov_token = "<OOV>"

    def fit_on_texts(self, texts, min_freq=1):
        """Build vocabulary from a list of URLs."""
        # Add special tokens
        self.char2idx[self.pad_token] = 0
        self.char2idx[self.oov_token] = 1
        
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)
            
        current_idx = 2
        for char, count in char_counts.items():
            if count >= min_freq:
                self.char2idx[char] = current_idx
                current_idx += 1
                
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        
    def encode(self, url):
        """Encode a single URL into a list of integers, padding/truncating as needed."""
        # Truncate
        url = url[:self.max_len]
        # Encode
        encoded = [self.char2idx.get(char, self.char2idx[self.oov_token]) for char in url]
        # Pad
        if len(encoded) < self.max_len:
            encoded = encoded + [self.char2idx[self.pad_token]] * (self.max_len - len(encoded))
        return encoded
        
    def batch_encode(self, list_of_urls):
        """Encode a batch of URLs."""
        return [self.encode(url) for url in list_of_urls]
        
    def save(self, path):
        """Save vocabulary to disk."""
        with open(path, 'w') as f:
            json.dump({
                "max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "char2idx": self.char2idx
            }, f, indent=4)
            
    def load(self, path):
        """Load vocabulary from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.max_len = data["max_len"]
            self.vocab_size = data["vocab_size"]
            self.char2idx = data["char2idx"]
            self.idx2char = {idx: char for char, idx in self.char2idx.items()}
