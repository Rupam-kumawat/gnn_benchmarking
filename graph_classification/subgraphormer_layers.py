

# subgraphormer_layers.py

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F

class Atom(nn.Module):
    """
    Atom encoder.
    MODIFIED: Now uses a standard Linear layer for non-OGB datasets
    to handle arbitrary node features, fixing the compatibility issue.
    """
    def __init__(self, dim: int, max_dis: int, use_linear: bool = False, atom_dim: int = 6, dataset_type: str = 'tu'):
        super().__init__()
        self.max_dis = max_dis
        
        # This is the key change:
        # Use the OGB-specific AtomEncoder only for 'ogb' datasets.
        # For all other datasets (like 'tu'), use a flexible Linear layer.
        self.dataset_type = dataset_type    
        if dataset_type == 'ogb' and not use_linear:
            self.embed_v = AtomEncoder(dim)
        else:
            self.embed_v = nn.Linear(atom_dim, dim)

        self.embed_d = nn.Embedding(max_dis + 2, dim)

    # def forward(self, batch):
    #     # Ensure the input tensor is float for the Linear layer
    #     x = self.embed_v(batch.x.to(torch.float))

    #     # Clamp distance values for the embedding layer
    #     d = torch.clamp(batch.d, max=self.max_dis)
    #     d[batch.d > 1000] = self.max_dis + 1 # Handle 'infinite' distance
    #     d = self.embed_d(d)
        
    #     batch.x = x + d
    #     del batch.d
    #     return batch
    
    def forward(self, batch):
        # ================================================================= #
        # THE FIX: Handle data types based on the dataset type              #
        # ================================================================= #
        if self.dataset_type == 'ogb':
            # OGB AtomEncoder expects integer categorical features
            x = self.embed_v(batch.x.to(torch.long))
        else:
            # TU datasets with continuous features need a float tensor
            x = self.embed_v(batch.x.to(torch.float))
        # ================================================================= #
        
        d = torch.clamp(batch.d, max=self.max_dis)
        d[batch.d > 1000] = self.max_dis + 1
        d = self.embed_d(d)
        
        batch.x = x + d
        del batch.d
        return batch

# ... (rest of the file remains the same)
class Bond(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = BondEncoder(dim)

    def forward(self, message, attrs):
        if attrs is None:
            return F.relu(message)
        return F.relu(message + self.embed(attrs))

class PE_layer(nn.Module):
    def __init__(self, num_eigen_vectors):
        super().__init__()
        self.num_eigen_vectors = num_eigen_vectors

    def forward(self, batch):
        sign_flip = (torch.randint(0, 2, (1, batch.subgraph_PE.size(1)), device=batch.x.device) * 2 - 1)
        PE_vector = batch.subgraph_PE * sign_flip
        PE_vector = PE_vector[:, :self.num_eigen_vectors]
        batch.x = torch.cat([batch.x, PE_vector], dim=-1)
        del batch.subgraph_PE
        return batch

class Attention_block(torch.nn.Module):
    def __init__(self, d, H=1, d_output=64, edge_dim=64, type='Gat'):
        super(Attention_block, self).__init__()
        self.H = H
        self.d_output = d_output
        self.edge_dim = edge_dim
        self.d = d
        if type == 'GatV2':
            self.attn_layer = gnn.GATv2Conv(in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        elif type == 'Transformer_conv':
            self.attn_layer = gnn.TransformerConv(in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        else:
            self.attn_layer = gnn.GATConv(in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.attn_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

class MLP(nn.Sequential):
    def __init__(self, idim: int, odim: int, hdim: int = None, norm: bool = True):
        super().__init__()
        hdim = hdim or idim
        self.add_module("input", nn.Linear(idim, hdim))
        if norm:
            self.add_module("input_bn", nn.BatchNorm1d(hdim))
        self.add_module("input_relu", nn.ReLU())
        self.add_module("output", nn.Linear(hdim, odim))

class Pooling(nn.Module):
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.predict = MLP(idim, odim, hdim=idim*2, norm=False)

    def forward(self, batch):
        return self.predict(gnn.global_mean_pool(batch.x, batch.batch))