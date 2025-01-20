import torch.nn as nn
from src.model.word_embedding import WordEmbedding
from experiment.model.positional_encoding_experiment import SinusoidalPositionalEncoding, LearnablePositionalEncoding

import torch
import torch.nn as nn
import math

# Modified Embedding class
class Embedding(nn.Module):
    """
    Combines token embeddings with positional encodings for input sequences.
    """
    def __init__(self, vocab_size, input_dim, max_len=5000, pos_encoding="sinusoidal"):
        """
        Initialize Embedding with token and positional embeddings.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            input_dim (int): Dimensionality of the embedding space.
            max_len (int): Maximum length of input sequences. Default is 5000.
            pos_encoding (str): Type of positional encoding ("sinusoidal" or "learnable")
        """
        super().__init__()
        self.token_embedding = WordEmbedding(vocab_size, input_dim)
        
        # Select positional encoding type
        if pos_encoding == "sinusoidal":
            self.position_encoding = SinusoidalPositionalEncoding(input_dim, max_len)
        elif pos_encoding == "learnable":
            self.position_encoding = LearnablePositionalEncoding(input_dim, max_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding}")

    def forward(self, tokens):
        """
        Forward pass to compute the combined embeddings.

        Args:
            tokens (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Combined token and positional embeddings with shape (batch_size, seq_len, d_model).
        """
        # Compute token embeddings
        token_emb = self.token_embedding(tokens)

        # Compute positional encodings
        pos_emb = self.position_encoding(token_emb)

        # Return the sum of token embeddings and positional encodings
        return token_emb + pos_emb
