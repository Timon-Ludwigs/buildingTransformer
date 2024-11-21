import torch
import torch.nn as nn
import math

# Implementation of the Input Embeddings

class InputEmbeddings(nn.modue):
    
    # d_model: dimension of the model
    # vocab_size: how many words are in the vocabulary

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # embedding layer for mapping words to vectors:
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # multiply by sqrt(d_model) as per the paper
    
# Implementation of the Positional Encoding

class PositionalEncoding(nn.Module):

    # d_model: dimension of the model
    # seq_length: length of the sequence
    # dropout: dropout rate to prevent overfitting

    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # Create matrix with shape (seq_length, d_model)
        # Need vectors of d_model size for each position in the sequence
        # Need seq_length number of these vectors, since the max. length of the sequence is seq_length
        pe = torch.zeros(seq_length, d_model)

        # Use a tensor that represents the word inside the sequence of shape (seq_length, 1)
        position = torch.arange(0, seq_length, dtype = torch.float).unsqueeze(1).float()

        # Implement the positional encoding functions
        # Sin used for even positions and Cos used for odd positions
        # Use exp(log()) for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # sin for even positions (start at 0, go forward by 2)
        pe[:, 0::2] = torch.sin(position * div_term)

        # cos for odd positions (start at 1, go forward by 2)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension to the tensor -> then we can apply it to all the sequeces
        pe = pe.unsqueeze(0) # pe has dimension of (1, seq_length, d_model)

        # register the tensor as a buffer
        # buffers won't be updated during training -> want to keep it in the model but not as a 
        # parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add positional encoding to every token
        # since the positional encodings are fixed, we dont want the model to learn them. So we set 
        # requires_grad to False
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)

        # apply dropout
        return self.dropout(x)

# Implementation of the Layer Normalization

# For batch j we want to implement:
# x^hat_j = (x_j - mu_j) / sqrt(sigma_j^2 + epsilon)

class LayerNormalization(nn.Module):

    def __init__(self, epsilon: float = 1e-10) -> None:
        super().__init__()

        # use epsilon for numerical stability (so we dont get too small or too large numbers) 
        # and to prevent division by zero
        self.epsilon = epsilon
        
        # alpha and beta are learnable parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # calculate mean and variance
        mu = x.mean(-1, keepdim = True)
        sigma = x.std(-1, keepdim = True)

        # normalize
        return self.alpha * (x - mu) / (sigma + self.epsilon) + self.beta
    
    # Implementation of the Feed Forward Network

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        # FFN conists of two linear transformations with a ReLU activation in between
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
