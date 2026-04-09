import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from .config import CONFIG
from .utils import get_logger, set_seed

logger = get_logger("DataLoader", log_file="logs/data_download.log")

class DatasetLoader:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.df = None
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    def download(self, primary_ds="ealvaradob/phishing-dataset"):
        """Attempt to load the advanced dataset from the parent directory, fallback to synthetic."""
        logger.info("Attempting to load advanced dataset...")
        dataset_path = os.path.abspath("../advanced_url_dataset.csv")
        try:
            if os.path.exists(dataset_path):
                logger.info(f"Found dataset at {dataset_path}, loading...")
                self.df = pd.read_csv(dataset_path)
                # Keep standard columns
                self.df = self.df[['url', 'label']]
                
                # Sample down to 20000 items to balance training time with data volume
                if len(self.df) > 20000:
                    self.df = self.df.sample(20000, random_state=42)
                
                self.df.to_csv(os.path.join(self.raw_dir, "dataset.csv"), index=False)
                logger.info("Successfully loaded primary dataset.")
            else:
                logger.error(f"Dataset not found at {dataset_path}. Proceeding with fallback to synthetic data.")
                self._create_synthetic_data()
            
        except Exception as e:
            logger.error(f"Dataset custom load failed: {e}")
            logger.info("Proceeding with fallback to synthetic data.")
            self._create_synthetic_data()

    def _create_synthetic_data(self):
        """Creates dummy synthetic balanced data when network/API fails."""
        logger.warning("Generating SYNTHETIC data as fallback!")
        
        benign_bases = [
            "https://www.google.com/search?q=",
            "http://github.com/user/repo_",
            "https://en.wikipedia.org/wiki/Page_",
            "https://stackoverflow.com/questions/",
            "http://example.com/index_",
            "https://www.amazon.com/dp/",
            "https://reddit.com/r/MachineLearning/comments/",
            "https://www.youtube.com/watch?v=",
            "https://news.ycombinator.com/item?id=",
            "http://www.bbc.co.uk/news/article_"
        ]
        
        phish_bases = [
            "http://secure-login-paypal.com.verify-update.info/login_",
            "http://appleid.apple.com.update.verification.cn/sign-in_",
            "http://www.netflix-cancel-account.com/billing_",
            "https://amazon-security-alert.net/confirm_",
            "http://chase.bank.verify.login-portal1.com/auth_",
            "http://192.168.1.100/admin/login.php?id=",
            "http://bit.ly/secure-auth-",
            "http://service-alert.microsoft.com.tk/login_",
            "http://wells-fargo-urgent.com/login_",
            "http://steam-community-free-games.com/login_"
        ]
        
        benign_urls = []
        phish_urls = []
        
        for i in range(100):
            for base in benign_bases:
                benign_urls.append(f"{base}{i}")
            for base in phish_bases:
                phish_urls.append(f"{base}{i}")
                
        urls = benign_urls + phish_urls
        labels = [0]*len(benign_urls) + [1]*len(phish_urls)
        
        self.df = pd.DataFrame({"url": urls, "label": labels})
        self.df.to_csv(os.path.join(self.raw_dir, "dataset.csv"), index=False)
        logger.info(f"Generated {len(self.df)} synthetic samples.")

    def load_raw(self):
        """Load raw dataset from disk."""
        path = os.path.join(self.raw_dir, "dataset.csv")
        if not os.path.exists(path):
            self.download()
        else:
            self.df = pd.read_csv(path)
        return self.df
        
    def clean(self):
        """Clean dataset (e.g., lowercasing, dropping nulls)."""
        logger.info("Cleaning data...")
        assert self.df is not None, "Dataset must be loaded before cleaning."
        self.df = self.df.dropna(subset=['url', 'label'])
        self.df['url'] = self.df['url'].astype(str).str.strip().str.lower()
        self.df['label'] = self.df['label'].astype(int)
        
    def dedupe(self):
        """Remove duplicate URLs."""
        assert self.df is not None, "Dataset must be loaded before deduplication."
        logger.info(f"Deduplicating... initial size: {len(self.df)}")
        self.df = self.df.drop_duplicates(subset=['url'])
        logger.info(f"Size after deduplication: {len(self.df)}")
        
    def extract_urls(self):
        """Placeholder for further feature extraction if needed."""
        # We are only using character-level models on the URL string.
        pass
        
    def split(self, seed=42, ratios=(0.8, 0.1, 0.1)):
        """Split data into train/val/test according to ratios."""
        logger.info("Splitting data...")
        set_seed(seed)
        train_ratio, val_ratio, test_ratio = ratios
        
        # Calculate val test ratios relative to the remainder
        temp_ratio = val_ratio + test_ratio
        val_rel_ratio = val_ratio / temp_ratio
        
        train_df, temp_df = train_test_split(self.df, test_size=temp_ratio, random_state=seed, stratify=self.df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_rel_ratio), random_state=seed, stratify=temp_df['label'])
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    def save_processed(self):
        """Save the processed splits."""
        logger.info("Saving processed datasets...")
        self.train_df.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
        self.val_df.to_csv(os.path.join(self.processed_dir, "val.csv"), index=False)
        self.test_df.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)
        logger.info("Processed datasets saved.")

if __name__ == "__main__":
    loader = DatasetLoader()
    loader.download()
    loader.clean()
    loader.dedupe()
    loader.split(seed=CONFIG["seed"], ratios=CONFIG["dataset_ratios"])
    loader.save_processed()
