import torch.nn as nn
from src.model.functional import TransformerDecoderLayer
from experiment.model.embedding_experiment import Embedding
from experiment.model.positional_encoding_experiment import SinusoidalPositionalEncoding, LearnablePositionalEncoding

class Decoder(nn.Module):
    def __init__(self, vocab_size, input_dim, max_len, num_heads, feature_dim, dropout, n_layers, pos_encoding="sinusoidal"):
        super().__init__()
        # Initialize embedding layer with specified positional encoding type
        self.embedding = Embedding(vocab_size, input_dim, max_len, pos_encoding)

        
        # Initialize positional encoding
        if pos_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(input_dim, max_len)
        else:  # learnable
            self.pos_encoding = LearnablePositionalEncoding(input_dim, max_len)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(input_dim, num_heads, feature_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, target_tokens, encoder_mask, decoder_mask):
        # Get embeddings and add positional encoding
        target_embeddings = self.embedding(target_tokens)
        target_embeddings = target_embeddings + self.pos_encoding(target_embeddings)
        target_embeddings = self.dropout(target_embeddings)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            target_embeddings = layer(
                encoder_outputs, target_embeddings, encoder_mask, decoder_mask
            )
        
        # Project to vocabulary space
        output_logits = self.output_projection(target_embeddings)
        return output_logits