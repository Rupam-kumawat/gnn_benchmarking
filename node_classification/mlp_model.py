# mlp_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class MLP(torch.nn.Module):
    """A simple two-layer Multi-Layer Perceptron."""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.get_embeddings(x)
        x = self.lin2(x)
        return x

    def get_embeddings(self, x):
        """Returns the output of the hidden layer."""
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x