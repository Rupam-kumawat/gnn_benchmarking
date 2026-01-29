

# FSGCN_models.py

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

__all__ = [
    "FSGNN",
    "FSGNN_Large"
]

class FSGNN(nn.Module):
    """
    The official FSGNN model with a learnable attention mechanism.
    Each pre-computed feature matrix is first passed through a dedicated linear layer,
    then weighted by an attention score, and finally concatenated.
    """
    def __init__(
        self, 
        nfeat: int, 
        nlayers: int, 
        nhidden: int, 
        nclass: int, 
        dropout: float,
        layer_norm: bool = False,
    ):
        super(FSGNN, self).__init__()
        self.fc2 = nn.Linear(nhidden * nlayers, nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        # A separate linear transformation for each feature matrix in the input list
        self.fc1 = nn.ModuleList([nn.Linear(nfeat, int(nhidden)) for _ in range(nlayers)])
        # Learnable attention parameter to weigh the importance of each feature matrix
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.layer_norm = layer_norm

    # ==================== INTEGRATION: METHOD ADDED ====================
    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for layer in self.fc1:
            layer.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.ones_(self.att)
    # =================================================================

    def forward(self, list_mat: list) -> torch.Tensor:
        """The forward pass for the FSGNN model."""
        # Get the embeddings from the hidden layer
        out = self.get_embeddings(list_mat)
        # Pass through the final classification layer
        out = self.fc2(out)
        return out

    # ==================== INTEGRATION: METHOD ADDED ====================
    def get_embeddings(self, list_mat: list) -> torch.Tensor:
        """
        Computes node embeddings from the hidden layer for visualization (e.g., t-SNE).
        This contains the core logic of the FSGNN model.
        """
        # Calculate attention weights by applying softmax to the learnable att parameter
        mask = self.sm(self.att)
        
        list_out = []
        for ind, mat in enumerate(list_mat):
            # Apply the specific linear layer for this feature matrix
            tmp_out = self.fc1[ind](mat)
            
            if self.layer_norm:
                # Optional L2 normalization
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            
            # Apply the calculated attention weight
            tmp_out = torch.mul(mask[ind], tmp_out)
            list_out.append(tmp_out)

        # Concatenate all the processed feature matrices
        final_mat = torch.cat(list_out, dim=1)
        
        # Apply activation and dropout
        out = self.act_fn(final_mat)
        out = F.dropout(out, self.dropout, training=self.training)
        
        return out
    # =================================================================

class FSGNN_Large(nn.Module):
    # Note: This model is designed for mini-batch training on very large graphs,
    # as indicated by the `st` (start) and `end` parameters in its forward pass.
    # It is not used by the current full-graph training pipeline in `main_with_fsgcn.py`.
    def __init__(
        self,
        nfeat,
        nlayers,
        nhidden,
        nclass,
        dp1,
        dp2,
        layer_norm: bool = True
    ):
        super(FSGNN_Large,self).__init__()
        self.wt1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.fc2 = nn.Linear(nhidden*nlayers,nhidden)
        self.fc3 = nn.Linear(nhidden,nclass)
        self.dropout1 = dp1 
        self.dropout2 = dp2 
        self.act_fn = nn.ReLU()
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.layer_norm = layer_norm

    def forward(self, list_adj, st=0, end=0):
        mask = self.sm(self.att)
        mask = torch.mul(len(list_adj),mask)

        list_out = list()
        for ind, mat in enumerate(list_adj):
            mat = mat[st:end,:].cuda()
            tmp_out = self.wt1[ind](mat)
            if self.layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)
            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout1,training=self.training)
        out = self.fc2(out)
        out = self.act_fn(out)
        out = F.dropout(out,self.dropout2,training=self.training)
        out = self.fc3(out)

        return out
