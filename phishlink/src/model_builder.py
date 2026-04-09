import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DPhishDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, max_len=200):
        super(CNN1DPhishDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D-CNN sequence:
        # Input shape to Conv1d should be (batch_size, embedding_dim, max_len)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=7, padding=3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128*3, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x is (batch_size, max_len)
        x = self.embedding(x)  # (batch_size, max_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, max_len)
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        # MaxPool across time dimension
        x1 = self.pool(x1).squeeze(-1) # (batch_size, 128)
        x2 = self.pool(x2).squeeze(-1)
        x3 = self.pool(x3).squeeze(-1)
        
        x = torch.cat([x1, x2, x3], dim=1) # (batch_size, 128*3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Logits (batch_size, 1)
        # return logits, sigmoid can be applied later optionally or by BCEWithLogitsLoss
        return x

class BiLSTMPhishDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=64, num_layers=2):
        super(BiLSTMPhishDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        # Extract last hidden state for both directions
        # hn shape: (num_layers*2, batch_size, hidden_size)
        forward_hn = hn[-2, :, :]
        backward_hn = hn[-1, :, :]
        
        x = torch.cat([forward_hn, backward_hn], dim=1)  # (batch_size, hidden_size*2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ModelBuilder:
    @staticmethod
    def build_cnn(num_chars, embedding_dim=64, max_len=200):
        return CNN1DPhishDetector(vocab_size=num_chars, embedding_dim=embedding_dim, max_len=max_len)
        
    @staticmethod
    def build_bilstm(num_chars, embedding_dim=64, max_len=200):
        return BiLSTMPhishDetector(vocab_size=num_chars, embedding_dim=embedding_dim)
