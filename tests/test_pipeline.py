import pytest
import torch
from anomaly.model import LSTMAutoencoder

def test_lstm_autoencoder_shapes():
    batch_size = 4
    seq_len = 10
    feature_dim = 9
    
    model = LSTMAutoencoder(input_size=feature_dim, hidden_size=32, num_layers=1)
    x = torch.randn(batch_size, seq_len, feature_dim)
    reconstructed = model(x)
    
    assert reconstructed.shape == (batch_size, seq_len, feature_dim)

def test_lstm_autoencoder_empty_forward():
    model = LSTMAutoencoder(input_size=9, hidden_size=16, num_layers=1)
    assert next(model.parameters()).is_cuda is False
