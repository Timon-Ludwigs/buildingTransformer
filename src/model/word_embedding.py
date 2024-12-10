import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize the Word Embedding layer.
        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimensionality of word embeddings.
        """
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Forward pass for word embeddings.
        :param x: Input tensor of word indices, shape (batch_size, seq_len).
        :return: Word embeddings, shape (batch_size, seq_len, embedding_dim).
        """
        return self.embedding(x)
