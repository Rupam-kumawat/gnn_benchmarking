# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv

# class GlobalAttn(torch.nn.Module):
#     def __init__(self, hidden_channels, heads, num_layers, beta, dropout, qk_shared=True):
#         super(GlobalAttn, self).__init__()

#         self.hidden_channels = hidden_channels
#         self.heads = heads
#         self.num_layers = num_layers
#         self.beta = beta
#         self.dropout = dropout
#         self.qk_shared = qk_shared

#         if self.beta < 0:
#             self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads*hidden_channels))
#         else:
#             self.betas = torch.nn.Parameter(torch.ones(num_layers, heads*hidden_channels)*self.beta)

#         self.h_lins = torch.nn.ModuleList()
#         if not self.qk_shared:
#             self.q_lins = torch.nn.ModuleList()
#         self.k_lins = torch.nn.ModuleList()
#         self.v_lins = torch.nn.ModuleList()
#         self.lns = torch.nn.ModuleList()
#         for i in range(num_layers):
#             self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
#             if not self.qk_shared:
#                 self.q_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
#             self.k_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
#             self.v_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
#             self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
#         self.lin_out = torch.nn.Linear(heads*hidden_channels, heads*hidden_channels)

#     def reset_parameters(self):
#         for h_lin in self.h_lins:
#             h_lin.reset_parameters()
#         if not self.qk_shared:
#             for q_lin in self.q_lins:
#                 q_lin.reset_parameters()
#         for k_lin in self.k_lins:
#             k_lin.reset_parameters()
#         for v_lin in self.v_lins:
#             v_lin.reset_parameters()
#         for ln in self.lns:
#             ln.reset_parameters()
#         if self.beta < 0:
#             torch.nn.init.xavier_normal_(self.betas)
#         else:
#             torch.nn.init.constant_(self.betas, self.beta)
#         self.lin_out.reset_parameters()

#     def forward(self, x):
#         seq_len, _ = x.size()
#         for i in range(self.num_layers):
#             h = self.h_lins[i](x)
#             k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
#             if self.qk_shared:
#                 q = k
#             else:
#                 q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
#             v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

#             # numerator
#             kv = torch.einsum('ndh, nmh -> dmh', k, v)
#             num = torch.einsum('ndh, dmh -> nmh', q, kv)

#             # denominator
#             k_sum = torch.einsum('ndh -> dh', k)
#             den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

#             # linear global attention based on kernel trick
#             if self.beta < 0:
#                 beta = F.sigmoid(self.betas[i]).unsqueeze(0)
#             else:
#                 beta = self.betas[i].unsqueeze(0)
#             x = (num/den).reshape(seq_len, -1)
#             x = self.lns[i](x) * (h+beta)
#             x = F.relu(self.lin_out(x))
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         return x



# class Polynormer(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, global_layers=2,
#             in_dropout=0.15, dropout=0.5, global_dropout=0.5, heads=1, beta=-1, pre_ln=False):
#         super(Polynormer, self).__init__()

#         self._global = False
#         self.in_drop = in_dropout
#         self.dropout = dropout
#         self.pre_ln = pre_ln

#         ## Two initialization strategies on beta
#         self.beta = beta
#         if self.beta < 0:
#             self.betas = torch.nn.Parameter(torch.zeros(local_layers,heads*hidden_channels))
#         else:
#             self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)

#         self.h_lins = torch.nn.ModuleList()
#         self.local_convs = torch.nn.ModuleList()
#         self.lins = torch.nn.ModuleList()
#         self.lns = torch.nn.ModuleList()
#         if self.pre_ln:
#             self.pre_lns = torch.nn.ModuleList()

#         for _ in range(local_layers):
#             self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
#             self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
#                 concat=True, add_self_loops=False, bias=False))
#             self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
#             self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
#             if self.pre_ln:
#                 self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))

#         self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
#         self.ln = torch.nn.LayerNorm(heads*hidden_channels)
#         self.global_attn = GlobalAttn(hidden_channels, heads, global_layers, beta, global_dropout)
#         self.pred_local = torch.nn.Linear(heads*hidden_channels, out_channels)
#         self.pred_global = torch.nn.Linear(heads*hidden_channels, out_channels)

#     def reset_parameters(self):
#         for local_conv in self.local_convs:
#             local_conv.reset_parameters()
#         for lin in self.lins:
#             lin.reset_parameters()
#         for h_lin in self.h_lins:
#             h_lin.reset_parameters()
#         for ln in self.lns:
#             ln.reset_parameters()
#         if self.pre_ln:
#             for p_ln in self.pre_lns:
#                 p_ln.reset_parameters()
#         self.lin_in.reset_parameters()
#         self.ln.reset_parameters()
#         self.global_attn.reset_parameters()
#         self.pred_local.reset_parameters()
#         self.pred_global.reset_parameters()
#         if self.beta < 0:
#             torch.nn.init.xavier_normal_(self.betas)
#         else:
#             torch.nn.init.constant_(self.betas, self.beta)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=self.in_drop, training=self.training)
#         x = self.lin_in(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         ## equivariant local attention
#         x_local = 0
#         for i, local_conv in enumerate(self.local_convs):
#             if self.pre_ln:
#                 x = self.pre_lns[i](x)
#             h = self.h_lins[i](x)
#             h = F.relu(h)
#             x = local_conv(x, edge_index) + self.lins[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             if self.beta < 0:
#                 beta = F.sigmoid(self.betas[i]).unsqueeze(0)
#             else:
#                 beta = self.betas[i].unsqueeze(0)
#             x = (1-beta)*self.lns[i](h*x) + beta*x
#             x_local = x_local + x

#         ## equivariant global attention
#         if self._global:
#             x_global = self.global_attn(self.ln(x_local))
#             x = self.pred_global(x_global)
#         else:
#             x = self.pred_local(x_local)

#         return x



# Save this file as polynormer_adapted.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
# from torch_scatter import scatter_add

# ===============================================================================================
# HELPER CLASSES AND FUNCTIONS FOR POLYNORMER (COPIED FROM ORIGINAL)
# ===============================================================================================

class LocalPropagation(MessagePassing):
    def __init__(self, in_channels, out_channels, pre_ln=False):
        super(LocalPropagation, self).__init__(aggr='add')
        self.pre_ln = pre_ln
        if self.pre_ln:
            self.ln = nn.LayerNorm(in_channels, elementwise_affine=False)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        if self.pre_ln:
            x = self.ln(x)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1).view(-1, 1)
        norm = torch.pow(deg, -0.5)
        norm_row = norm[row]
        norm_col = norm[col]

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, norm_row=norm_row, norm_col=norm_col)
        return out

    def message(self, x_j, norm_row, norm_col):
        return norm_row * x_j * norm_col

    def reset_parameters(self):
        if self.pre_ln:
            self.ln.reset_parameters()
        self.lin.reset_parameters()

class GlobalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads, beta=-1.0, dropout=0.0):
        super(GlobalAttention, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_dropout = nn.Dropout(dropout)
        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_final = nn.Linear(heads * out_channels, out_channels)
        # self.beta = beta
        # Inside the __init__ of GlobalAttention
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        n = x.size(0)
        q = self.lin_q(x).view(n, self.heads, self.out_channels)
        k = self.lin_k(x).view(n, self.heads, self.out_channels)
        v = self.lin_v(x).view(n, self.heads, self.out_channels)
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        
        if self.beta < 0:
            beta = F.softplus(self.beta)
        else:
            beta = self.beta
            
        alpha = beta * (q @ k.transpose(-2, -1)) / (self.out_channels ** 0.5)
        alpha = alpha.softmax(dim=-1)
        alpha = self.att_dropout(alpha)
        
        v = v.transpose(0, 1)
        out = alpha @ v
        out = out.transpose(0, 1).contiguous().view(n, -1)
        out = self.lin_final(out)
        return out

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()
        self.lin_final.reset_parameters()

# ===============================================================================================
# INTERNAL POLYNORMER CLASS (REFACTORED WITH get_embeddings)
# ===============================================================================================

class _Polynormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=2, 
                 global_layers=2, heads=4, in_dropout=0.0, dropout=0.0,
                 global_dropout=0.0, beta=-1.0, pre_ln=False):
        super(_Polynormer, self).__init__()
        self.in_dropout = nn.Dropout(in_dropout)
        self.dropout = nn.Dropout(dropout)
        self.local_layers = local_layers
        self.global_layers = global_layers
        self.pre_ln = pre_ln
        
        self.local_props = nn.ModuleList()
        self.global_atts = nn.ModuleList()
        self.local_norms = nn.ModuleList()
        self.global_norms = nn.ModuleList()

        # Input layer
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        
        # Local and global layers
        for _ in range(local_layers):
            self.local_props.append(LocalPropagation(hidden_channels, hidden_channels, pre_ln))
            self.local_norms.append(nn.LayerNorm(hidden_channels))
        
        for _ in range(global_layers):
            self.global_atts.append(GlobalAttention(hidden_channels, hidden_channels // heads, heads, beta, global_dropout))
            self.global_norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        for prop in self.local_props:
            prop.reset_parameters()
        for norm in self.local_norms:
            norm.reset_parameters()
        for att in self.global_atts:
            att.reset_parameters()
        for norm in self.global_norms:
            norm.reset_parameters()
        self.lin_out.reset_parameters()

    def get_embeddings(self, x, edge_index):
        """
        This new method contains the core logic for generating node embeddings.
        """
        x = self.in_dropout(x)
        x = self.lin_in(x)

        for i in range(self.local_layers):
            x_res = x
            x = self.local_props[i](x, edge_index)
            x = self.dropout(x)
            x = self.local_norms[i](x + x_res)
        
        for i in range(self.global_layers):
            x_res = x
            x = self.global_atts[i](x)
            x = self.dropout(x)
            x = self.global_norms[i](x + x_res)
            
        return x

    def forward(self, x, edge_index):
        """
        The forward pass is now simpler, using get_embeddings.
        """
        x_emb = self.get_embeddings(x, edge_index)
        x_out = self.lin_out(x_emb)
        return x_out

# ===============================================================================================
# THE NEW WRAPPER CLASS (This is the one you should import and use)
# ===============================================================================================

class PolynormerAdapted(torch.nn.Module):
    """
    A wrapper for the Polynormer model to make it compatible with the GNN training pipeline.
    It mimics the interface of the MPNNs class.
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 local_layers=2, global_layers=2, # Added global_layers for flexibility
                 dropout=0.5, heads=4, pre_ln=False,
                 # The following arguments from MPNNs are not used by Polynormer
                 # but are kept here for compatibility if your script passes them.
                 pre_linear=None, res=None, ln=None, bn=None, jk=None, gnn=None):
        
        super(PolynormerAdapted, self).__init__()

        # Here, we instantiate the actual _Polynormer model, mapping the arguments.
        self.polynormer = _Polynormer(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            local_layers=local_layers,
            global_layers=global_layers,
            heads=heads,
            in_dropout=dropout, # Map main dropout to in_dropout
            dropout=dropout,
            global_dropout=dropout, # Using the same dropout for all
            pre_ln=pre_ln
        )

    def reset_parameters(self):
        self.polynormer.reset_parameters()

    def get_embeddings(self, x, edge_index):
        return self.polynormer.get_embeddings(x, edge_index)

    def forward(self, x, edge_index):
        return self.polynormer.forward(x, edge_index)