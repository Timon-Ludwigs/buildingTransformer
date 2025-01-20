import torch.nn as nn
from src.model.functional import TransformerEncoderLayer
from experiment.model.positional_encoding_experiment import SinusoidalPositionalEncoding, LearnablePositionalEncoding
from experiment.model.embedding_experiment import Embedding

class Encoder(nn.Module):
    def __init__(self, vocab_size, input_dim, max_len, num_heads, feature_dim, dropout, n_layers, pos_encoding="sinusoidal"):
        super().__init__()
        # Initialize embedding layer without positional encoding
        self.embedding = Embedding(vocab_size, input_dim)
        
        # Initialize positional encoding
        if pos_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(input_dim, max_len)
        else:  # learnable
            self.pos_encoding = LearnablePositionalEncoding(input_dim, max_len)
        
        # Create encoder layers
        self.encoder = nn.Sequential(
            *[TransformerEncoderLayer(input_dim, num_heads, feature_dim, dropout)
              for _ in range(n_layers)]
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_token, src_mask):
        # Get embeddings and add positional encoding
        embedded = self.embedding(src_token)
        embedded = embedded + self.pos_encoding(embedded)
        embedded = self.dropout(embedded)
        
        # Apply encoder layers
        output = embedded
        for layer in self.encoder:
            output = layer(output, src_mask)
        
        return output