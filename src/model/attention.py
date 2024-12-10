import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, mask_future: bool = False):
        """
        Initialize the Attention layer.
        :param mask_future: If True, applies future masking for causal attention.
        """
        super(Attention, self).__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, attention_mask=None):
        """
        Compute attention scores and apply them to the value tensor.
        :param query: Tensor of shape (batch_size, seq_len_query, dim).
        :param key: Tensor of shape (batch_size, seq_len_key, dim).
        :param value: Tensor of shape (batch_size, seq_len_key, dim).
        :param attention_mask: Tensor of shape (batch_size, seq_len_query, seq_len_key) 
                               or (batch_size, seq_len_key) for value masking.
        :return: Attention-weighted values.
        """
        # Compute attention scores: Q * K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))

        # Apply future mask if specified (only for self-attention)
        if self.mask_future:
            seq_len_query, seq_len_key = scores.size(-2), scores.size(-1)
            future_mask = torch.triu(torch.ones(seq_len_query, seq_len_key), diagonal=1).to(scores.device).bool()
            scores = scores.masked_fill(future_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Align dimensions of mask with scores
            if attention_mask.dim() == 2:  # Value mask
                attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len_key)
            elif attention_mask.dim() == 3:  # Query-Key mask
                attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len_query, seq_len_key)

            # Broadcast to scores shape
            attention_mask = attention_mask.expand_as(scores)  # (batch_size, seq_len_query, seq_len_key)
            scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))

        # Compute softmax over scores
        attention_weights = F.softmax(scores, dim=-1)

        # Compute output: Attention weights * Value
        output = torch.matmul(attention_weights, value)

        return output