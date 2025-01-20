import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from experiment.model.encoder_experiment import Encoder
from experiment.model.decoder_experiment import Decoder
import math
from experiment.model.positional_encoding_experiment import LearnablePositionalEncoding, SinusoidalPositionalEncoding

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
    pos_encoding: str = "sinusoidal"  # "sinusoidal" or "learnable"

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
            'max_len': lambda x: x > 0,
            'pos_encoding': lambda x: x in ["sinusoidal", "learnable"]
        }
        for field, condition in validators.items():
            value = getattr(self, field)
            if not condition(value):
                raise ValueError(f"{field} has an invalid value: {value}")


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, input_dim, num_heads, num_encoder_layers, 
                 num_decoder_layers, feature_dim, dropout, max_len, 
                 pos_encoding="sinusoidal"):
        super().__init__()

        # Initialize encoder and decoder with specified positional encoding type
        self.encoder = Encoder(
            vocab_size=vocab_size,
            input_dim=input_dim,
            max_len=max_len,
            num_heads=num_heads,
            feature_dim=feature_dim,
            dropout=dropout,
            n_layers=num_encoder_layers,
            pos_encoding=pos_encoding
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            input_dim=input_dim,
            max_len=max_len,
            num_heads=num_heads,
            feature_dim=feature_dim,
            dropout=dropout,
            n_layers=num_decoder_layers,
            pos_encoding=pos_encoding
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.encoder(src, src_mask)
        return self.decoder(src_emb, tgt, src_mask, tgt_mask)

def create_experiment_configs():
    base_config = {
        "vocab_size": 50000,
        "input_dim": 64,
        "num_heads": 2,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "feature_dim": 64,
        "dropout": 0.00001,
        "max_len": 64,
    }
    
    sinusoidal_config = {**base_config, "pos_encoding": "sinusoidal"}
    learnable_config = {**base_config, "pos_encoding": "learnable"}
    
    return sinusoidal_config, learnable_config

def compare_models(src, tgt, src_mask, tgt_mask):
    sinusoidal_config, learnable_config = create_experiment_configs()
    
    # Initialize models
    sinusoidal_model = TransformerModel(**sinusoidal_config)
    learnable_model = TransformerModel(**learnable_config)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    sinusoidal_optimizer = torch.optim.Adam(sinusoidal_model.parameters())
    learnable_optimizer = torch.optim.Adam(learnable_model.parameters())
    
    # Forward pass and loss computation
    sinusoidal_output = sinusoidal_model(src, tgt, src_mask, tgt_mask)
    learnable_output = learnable_model(src, tgt, src_mask, tgt_mask)
    
    return sinusoidal_model, learnable_model

def main():
    # Test data
    src = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    tgt = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    src_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
    tgt_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
    
    sinusoidal_model, learnable_model = compare_models(src, tgt, src_mask, tgt_mask)
    print("Models created successfully!")

if __name__ == "__main__":
    main()