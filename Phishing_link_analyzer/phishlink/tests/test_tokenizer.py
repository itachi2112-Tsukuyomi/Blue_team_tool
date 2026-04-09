import os
import sys
import pytest

# Add parent directory to sys.path to resolve 'src' imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tokenizer import URLTokenizer

def test_tokenizer_fit():
    tokenizer = URLTokenizer(max_len=10)
    urls = ["http://test.com", "https://example.org"]
    tokenizer.fit_on_texts(urls, min_freq=1)
    
    assert tokenizer.vocab_size > 2  # PAD and OOV + characters
    assert tokenizer.pad_token in tokenizer.char2idx
    assert tokenizer.oov_token in tokenizer.char2idx

def test_tokenizer_encode():
    tokenizer = URLTokenizer(max_len=10)
    urls = ["abc"]
    tokenizer.fit_on_texts(urls, min_freq=1)
    
    encoded = tokenizer.encode("abx")  # x is OOV
    assert len(encoded) == 10
    assert encoded[0] == tokenizer.char2idx['a']
    assert encoded[1] == tokenizer.char2idx['b']
    assert encoded[2] == tokenizer.char2idx[tokenizer.oov_token]
    assert encoded[3] == tokenizer.char2idx[tokenizer.pad_token]  # padded

def test_tokenizer_save_load(tmp_path):
    tokenizer = URLTokenizer(max_len=10)
    urls = ["abc"]
    tokenizer.fit_on_texts(urls)
    
    save_path = tmp_path / "vocab.json"
    tokenizer.save(str(save_path))
    
    new_tokenizer = URLTokenizer(max_len=5) # Wrong max len initially
    new_tokenizer.load(str(save_path))
    
    assert new_tokenizer.max_len == 10
    assert new_tokenizer.vocab_size == tokenizer.vocab_size
    assert new_tokenizer.char2idx == tokenizer.char2idx
