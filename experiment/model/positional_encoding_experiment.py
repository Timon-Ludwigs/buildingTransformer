import torch
import torch.nn as nn
import math

class LearnablePositionalEncoding(nn.Module):
    """
    Implements a learnable positional encoding using parameter matrix.
    """
    def __init__(self, input_dim, max_len=5000):
        super().__init__()
        
        # Create learnable parameter matrix
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, input_dim))
        
        # Initialize with small random values
        nn.init.xavier_uniform_(self.pos_embedding)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim)
        Returns:
            torch.Tensor: Positional encodings (1, seq_len, input_dim)
        """
        return self.pos_embedding[:, :x.size(1)]

class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from the 'Attention Is All You Need' paper.
    """
    def __init__(self, input_dim, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(1, max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) * -(math.log(10000.0) / input_dim))
        
        # Calculate sinusoidal pattern
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim)
        Returns:
            torch.Tensor: Positional encodings (1, seq_len, input_dim)
        """
        return self.pe[:, :x.size(1)]