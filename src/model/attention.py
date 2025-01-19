import torch
import torch.nn as nn
import torch.nn.functional as F


# class Attention(nn.Module):
#     def __init__(self, mask_future: bool = False):
#         super(Attention, self).__init__()
#         self.mask_future = mask_future

#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Implements the attention mechanism.

#         Args:
#             query: Tensor of shape (batch_size, seq_len_q, d_model)
#             key: Tensor of shape (batch_size, seq_len_k, d_model)
#             value: Tensor of shape (batch_size, seq_len_v, d_model)
#             attention_mask: Tensor of shape (batch_size, seq_len_q, seq_len_k),
#                             where 1 indicates valid positions and 0 indicates masked positions.

#         Returns:
#             Tensor of shape (batch_size, seq_len_q, d_model) containing the attention outputs.
#         """
#         # Compute attention scores
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

#         # Apply attention mask
#         if attention_mask is not None:
#             scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

#         # Optionally mask future positions
#         if self.mask_future:
#             seq_len = query.size(1)
#             future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
#             scores = scores.masked_fill(future_mask.to(query.device), float('-inf'))

#         # Compute attention weights
#         attention_weights = F.softmax(scores, dim=-1)

#         # Compute the output
#         output = torch.matmul(attention_weights, value)
#         return output

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, mask_future: bool = False):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         self.query_transform = nn.Linear(d_model, d_model, bias=False)
#         self.key_transform = nn.Linear(d_model, d_model, bias=False)
#         self.value_transform = nn.Linear(d_model, d_model, bias=False)
#         self.output_transform = nn.Linear(d_model, d_model, bias=False)

#         self.attention = Attention(mask_future)

#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Implements the multi-head attention mechanism.

#         Args:
#             query: Tensor of shape (batch_size, seq_len_q, d_model)
#             key: Tensor of shape (batch_size, seq_len_k, d_model)
#             value: Tensor of shape (batch_size, seq_len_v, d_model)
#             attention_mask: Tensor of shape (batch_size, seq_len_q, seq_len_k),
#                             where 1 indicates valid positions and 0 indicates masked positions.

#         Returns:
#             Tensor of shape (batch_size, seq_len_q, d_model) containing the multi-head attention outputs.
#         """
#         batch_size = query.size(0)

#         # Linear transformations and split into heads
#         query = self.query_transform(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         key = self.key_transform(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         value = self.value_transform(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

#         # Apply attention mechanism
#         attention_mask = attention_mask.unsqueeze(1)  # Expand mask for multiple heads
#         attention_output = self.attention(query, key, value, attention_mask)

#         # Concatenate heads and apply final linear transformation
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
#         output = self.output_transform(attention_output)

#         return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, mask_future=False):
        """
        Initialize Attention layer.
        
        Args:
            mask_future (bool): If True, masks future positions in self-attention
        """
        super().__init__()
        self.mask_future = mask_future
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Compute attention.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            attention_mask (torch.Tensor, optional): Mask to apply to attention scores
        
        Returns:
            torch.Tensor: Attention output
        """
        # Compute dot product attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        
        # Apply future masking if specified
        if self.mask_future and query.size(-2) > 1:
            # Create a lower triangular mask
            mask = torch.tril(torch.ones(query.size(-2), query.size(-2), 
                                         device=query.device)).bool()
            masked_scores = scores.masked_fill(~mask, float('-inf'))
        else:
            masked_scores = scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create a mask that's broadcastable to the scores
            mask = attention_mask.unsqueeze(-2).expand_as(masked_scores)
            masked_scores = masked_scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Compute softmax attention weights
        attention_weights = F.softmax(masked_scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, value)
        
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mask_future=False):
        """
        Initialize Multi-Head Attention layer.
        
        Args:
            embed_dim (int): Dimensionality of input embeddings
            num_heads (int): Number of attention heads
            mask_future (bool): If True, masks future positions in self-attention
        """
        super().__init__()
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mask_future = mask_future
        
        # Transformation matrices for query, key, and value
        self.query_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output transformation
        self.output_transform = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def _split_heads(self, x):
        """
        Split input into multiple heads.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Tensor split into multiple heads
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _combine_heads(self, x):
        """
        Combine multiple heads back to original dimension.
        
        Args:
            x (torch.Tensor): Multi-head tensor
        
        Returns:
            torch.Tensor: Combined tensor
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Compute multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            attention_mask (torch.Tensor, optional): Mask to apply to attention scores
        
        Returns:
            torch.Tensor: Multi-head attention output
        """
        # Transform and split heads
        Q = self._split_heads(self.query_transform(query))
        K = self._split_heads(self.key_transform(key))
        V = self._split_heads(self.value_transform(value))
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply future masking if specified
        if self.mask_future and query.size(-2) > 1:
            # Create a lower triangular mask
            mask = torch.tril(torch.ones(query.size(-2), query.size(-2), 
                                         device=query.device)).bool()
            masked_scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            masked_scores = scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create a mask that's broadcastable to the scores
            mask = attention_mask.unsqueeze(1).unsqueeze(1).expand_as(masked_scores)
            masked_scores = masked_scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Compute softmax attention weights
        attention_weights = F.softmax(masked_scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Combine heads and apply output transformation
        output = self.output_transform(self._combine_heads(context))
        
        return output