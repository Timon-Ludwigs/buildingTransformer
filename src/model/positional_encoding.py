import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int):
        """
        Initialize the Positional Encoding layer.
        :param embedding_dim: Dimensionality of word embeddings.
        :param max_seq_len: Maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()
        # Create the positional encoding matrix
        position = torch.arange(0, max_seq_len).unsqueeze(1)  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embedding_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_len, embedding_dim).
        :return: Tensor with positional encoding added, same shape as x.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]