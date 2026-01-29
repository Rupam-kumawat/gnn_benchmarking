
    
    
    
# pe_layer.py

import torch
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigsh

class LapPENodeEncoder(torch.nn.Module):
    """
    Laplacian Positional Embedding node encoder.
    This is a faithful replication of the LapPE logic from the official GraphGPS repository.
   
    """
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        self.dim_emb = dim_emb
        self.expand_x = expand_x
        self.linear_pe = nn.Linear(2 * dim_emb, dim_emb)

    def forward(self, data):
        if not hasattr(data, 'EigVals') or not hasattr(data, 'EigVecs'):
            raise ValueError("Pre-computed Laplacian PEs are missing from the data object.")

        lap_pe = torch.cat(
            [data.EigVecs.view(-1, self.dim_emb), data.EigVals.view(-1, self.dim_emb)], dim=-1
        )
        lap_pe = self.linear_pe(lap_pe)

        if self.expand_x:
            data.x = data.x + lap_pe
        else:
            data.x = torch.cat((data.x, lap_pe), -1)

        return data

def compute_laplacian_pe(data, dim_pe):
    """
    Pre-computes the Laplacian Positional Encodings for a single graph.
    """
    num_nodes = data.num_nodes
    if num_nodes <= dim_pe:
        data.EigVals = torch.zeros(num_nodes, dim_pe, dtype=torch.float)
        data.EigVecs = torch.zeros(num_nodes, dim_pe, dtype=torch.float)
        return data

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    eig_vals, eig_vecs = eigsh(adj, k=dim_pe + 1, which='SA', tol=1e-2, v0=np.ones(adj.shape[0]))
    
    eig_vecs = torch.from_numpy(eig_vecs[:, 1:]).float()
    eig_vals = torch.from_numpy(eig_vals[1:]).float().unsqueeze(0).repeat(num_nodes, 1)

    data.EigVals = eig_vals
    data.EigVecs = eig_vecs
    
    return data