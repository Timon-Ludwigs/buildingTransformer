import torch
import torch.nn as nn

    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return self.dropout(x)
