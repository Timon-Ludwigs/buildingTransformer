import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from src.model.encoder import Encoder
from src.model.decoder import Decoder

@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    vocab_size: int
    input_dim: int
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    feature_dim: int
    dropout: float
    max_len: int

    def validate(self):
        """Validate the configuration to ensure correctness."""
        validators = {
            'vocab_size': lambda x: x > 0,
            'input_dim': lambda x: x > 0,
            'num_heads': lambda x: x > 0,
            'num_encoder_layers': lambda x: x > 0,
            'num_decoder_layers': lambda x: x > 0,
            'feature_dim': lambda x: x > 0,
            'dropout': lambda x: 0 <= x <= 1,
            'max_len': lambda x: x > 0
        }
        for field, condition in validators.items():
            value = getattr(self, field)
            if not condition(value):
                raise ValueError(f"{field} has an invalid value: {value}")

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, input_dim, num_heads, num_encoder_layers, num_decoder_layers, feature_dim, dropout, max_len):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(
            vocab_size=vocab_size,
            input_dim=input_dim,
            max_len=max_len,
            num_heads=num_heads,
            feature_dim=feature_dim,
            dropout=dropout,
            n_layers=num_encoder_layers,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            input_dim=input_dim,
            max_len=max_len,
            num_heads=num_heads,
            feature_dim=feature_dim,
            dropout=dropout,
            n_layers=num_decoder_layers,
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.encoder(src, src_mask)
        return self.decoder(src_emb, tgt, src_mask, tgt_mask)

def main():
    # Configuration parameters
    config = {
        "vocab_size": 50000,
        "input_dim": 64,
        "num_heads": 2,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "feature_dim": 64,
        "dropout": 0.00001,
        "max_len": 64,
    }

    # Test data
    src = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    tgt = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    src_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
    tgt_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])

    # Initialize Transformer model with configuration
    model = TransformerModel(**config)
    print(model)

    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    probs = F.softmax(output, dim=-1)

    # Get most likely token indices
    token_ids = torch.argmax(probs, dim=-1)
    print("Token IDs:", token_ids)

if __name__ == "__main__":
    main()