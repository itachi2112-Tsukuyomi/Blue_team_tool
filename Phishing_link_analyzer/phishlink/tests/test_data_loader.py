import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import DatasetLoader

@pytest.fixture
def loader():
    return DatasetLoader(raw_dir="data/test_raw", processed_dir="data/test_processed")

def test_data_loader_synthetic(loader):
    """Test the synthetic data generation and processing pipeline."""
    # Force synthetic data generation
    loader._create_synthetic_data()
    
    assert loader.df is not None
    assert len(loader.df) == 2000
    assert 'url' in loader.df.columns
    assert 'label' in loader.df.columns
    
    # Test cleaning
    loader.clean()
    assert loader.df['url'].str.islower().all()
    
    # Test deduplication
    original_len = len(loader.df)
    loader.dedupe()
    deduped_len = len(loader.df)
    # Since synthetic data contains duplicates by design (10 unique repeated 100 times)
    assert deduped_len == 20
    
    # Test splitting
    loader.split(seed=42, ratios=(0.8, 0.1, 0.1))
    
    assert len(loader.train_df) == 16  # 80% of 20
    assert len(loader.val_df) == 2     # 10% of 20
    assert len(loader.test_df) == 2    # 10% of 20
    
    # Test saving
    loader.save_processed()
    assert os.path.exists("data/test_processed/train.csv")
    assert os.path.exists("data/test_processed/val.csv")
    assert os.path.exists("data/test_processed/test.csv")
    
    # Cleanup
    import shutil
    shutil.rmtree("data/test_raw", ignore_errors=True)
    shutil.rmtree("data/test_processed", ignore_errors=True)
