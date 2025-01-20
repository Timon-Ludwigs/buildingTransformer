import torch.nn as nn
from src.model.functional import TransformerEncoderLayer
from src.model.embedding import Embedding

class Encoder(nn.Module):
    def __init__(self, vocab_size, input_dim, max_len, num_heads, feature_dim, dropout, n_layers):
        super(Encoder, self).__init__()
        # Initialize embedding layer
        self.embedding = Embedding(vocab_size, input_dim, max_len)
        
        # Create encoder layers using nn.Sequential for streamlined execution
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(input_dim, num_heads, feature_dim, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_token, src_mask):
        # Embed input tokens
        embedded_tokens = self.embedding(src_token)
        
        # Apply the encoder layers
        output = embedded_tokens
        for layer in self.encoder:
            output = layer(output, src_mask)

        return output