import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, input_dim, max_len):
        """
        constructor of sinusoid encoding class

        :param input_dim: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, input_dim)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(
            0,
            input_dim,
            step=2,
        ).float()
        # 'i' means index of input_dim (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / input_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / input_dim)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # x.size(1) -> Sequence lenght

        return self.encoding[: x.size(1), :]
