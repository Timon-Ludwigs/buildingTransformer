import torch.nn as nn
from src.model.functional import TransformerDecoderLayer
from src.model.embedding import Embedding

class Decoder(nn.Module):
    """
    Transformer Decoder consisting of embedding layers, multiple decoder layers, and an output layer.
    """

    def __init__(self, vocab_size, input_dim, max_len, num_heads, feature_dim, dropout, n_layers):
        """
        Initialize the Transformer Decoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            input_dim (int): Dimensionality of embeddings and hidden states.
            max_len (int): Maximum length of input sequences.
            num_heads (int): Number of attention heads in each decoder layer.
            feature_dim (int): Dimensionality of the feedforward layer in the decoder.
            dropout (float): Dropout rate for regularization.
            n_layers (int): Number of decoder layers.
        """
        super(Decoder, self).__init__()
        self.embedding_layer = Embedding(vocab_size, input_dim, max_len)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(input_dim, num_heads, feature_dim, dropout)
            for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(input_dim, vocab_size)

    def forward(self, encoder_outputs, target_tokens, encoder_mask, decoder_mask):
        """
        Forward pass through the Transformer Decoder.

        Args:
            encoder_outputs (torch.Tensor): Output tensor from the encoder with shape (batch_size, src_seq_len, input_dim).
            target_tokens (torch.Tensor): Input target tensor with shape (batch_size, tgt_seq_len).
            encoder_mask (torch.Tensor): Mask for the encoder output with shape (batch_size, 1, src_seq_len).
            decoder_mask (torch.Tensor): Mask for the target input with shape (batch_size, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output logits for each vocabulary token with shape (batch_size, tgt_seq_len, vocab_size).
        """
        # Compute embeddings for target tokens
        target_embeddings = self.embedding_layer(target_tokens)

        # Pass through each decoder layer
        for layer in self.decoder_layers:
            target_embeddings = layer(
                encoder_outputs, target_embeddings, encoder_mask, decoder_mask
            )

        # Project decoder outputs to vocabulary space
        output_logits = self.output_projection(target_embeddings)
        return output_logits