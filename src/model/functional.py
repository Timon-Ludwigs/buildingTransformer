import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import MultiHeadAttention
from src.model.position_wise_FFN import PositionWiseFeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.feature_transformation = PositionWiseFeedForward(input_dim, feature_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, attention_mask)
        x = self.layer_norm_1(x + attn_output)
        x = self.dropout(x)

        # Feed-forward
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_2(x + ff_output)
        x = self.dropout(x)

        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads)
        self.feature_transformation = PositionWiseFeedForward(
            input_dim, feature_dim, dropout
        )

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, source_embedding, target_sequence, memory_mask=None, target_mask=None):
        # Self-attention layer
        self_attention_output = self.self_attention(target_sequence, target_sequence, target_sequence, target_mask)
        target_sequence = target_sequence + self.dropout(self_attention_output)
        target_sequence = self.layer_norm_1(target_sequence)

        # Cross-attention layer
        cross_attention_output = self.encoder_attention(
            target_sequence, source_embedding, source_embedding, memory_mask
        )
        target_sequence = target_sequence + self.dropout(cross_attention_output)
        target_sequence = self.layer_norm_2(target_sequence)

        # Feed-forward layer
        ff_output = self.feature_transformation(target_sequence)
        target_sequence = target_sequence + self.dropout(ff_output)
        target_sequence = self.layer_norm_3(target_sequence)

        return target_sequence