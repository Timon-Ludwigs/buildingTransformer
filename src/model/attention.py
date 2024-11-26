import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Perform scaled dot-product attention.

        Args:
            query: Tensor of shape (batch_size, num_heads, seq_len, d_k)
            key: Tensor of shape (batch_size, num_heads, seq_len, d_k)
            value: Tensor of shape (batch_size, num_heads, seq_len, d_v)
            mask: Tensor of shape (batch_size, 1, 1, seq_len) to mask out attention scores
        
        Returns:
            output: Tensor of shape (batch_size, num_heads, seq_len, d_v)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Calculate the attention scores (QK^T / sqrt(d_k))
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute the final output
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        output, attention_weights = self.attention(query, key, value, mask)
        
        # Concatenate heads and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out_linear(output)
        return self.dropout(output), attention_weights