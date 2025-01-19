import torch.nn as nn
from src.model.word_embedding import WordEmbedding
from src.model.positional_encoding import PositionalEncoding

class Embedding(nn.Module):
    """
    Combines token embeddings with positional encodings for input sequences.
    """

    def __init__(self, vocab_size, input_dim, max_len=5000):
        """
        Initialize Embedding with token and positional embeddings.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            input_dim (int): Dimensionality of the embedding space.
            max_len (int): Maximum length of input sequences. Default is 5000.
        """
        super(Embedding, self).__init__()
        self.token_embedding = WordEmbedding(vocab_size, input_dim)
        self.position_encoding = PositionalEncoding(input_dim, max_len)

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
        pos_emb = self.position_encoding(tokens)

        # Return the sum of token embeddings and positional encodings
        return token_emb + pos_emb
