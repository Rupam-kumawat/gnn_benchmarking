# import math
# import os
# from multiprocessing.sharedctypes import Value

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
# from torch_geometric.utils import degree
# from torch_sparse import SparseTensor, matmul

# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
#                  dropout=0.5, save_mem=True, use_bn=True):
#         super(GCN, self).__init__()

#         self.convs = nn.ModuleList()
#         # self.convs.append(
#         #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
#         self.convs.append(
#             GCNConv(in_channels, hidden_channels, cached=not save_mem))

#         self.bns = nn.ModuleList()
#         self.bns.append(nn.BatchNorm1d(hidden_channels))
#         for _ in range(num_layers - 2):
#             # self.convs.append(
#             #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
#             self.convs.append(
#                 GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
#             self.bns.append(nn.BatchNorm1d(hidden_channels))

#         # self.convs.append(
#         #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
#         self.convs.append(
#             GCNConv(hidden_channels, out_channels, cached=not save_mem))

#         self.dropout = dropout
#         self.activation = F.relu
#         self.use_bn = use_bn

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x, edge_index):
#         edge_weight=None
#         for i, conv in enumerate(self.convs[:-1]):
#             if edge_weight is None:
#                 x = conv(x, edge_index)
#             else:
#                 x=conv(x,edge_index,edge_weight)
#             if self.use_bn:
#                 x = self.bns[i](x)
#             x = self.activation(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return x
    
# def full_attention_conv(qs, ks, vs, output_attn=False):
#     # normalize input
#     qs = qs / torch.norm(qs, p=2)  # [N, H, M]
#     ks = ks / torch.norm(ks, p=2)  # [L, H, M]
#     N = qs.shape[0]

#     # numerator
#     kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
#     attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
#     attention_num += N * vs

#     # denominator
#     all_ones = torch.ones([ks.shape[0]]).to(ks.device)
#     ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
#     attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

#     # attentive aggregated results
#     attention_normalizer = torch.unsqueeze(
#         attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
#     attention_normalizer += torch.ones_like(attention_normalizer) * N
#     attn_output = attention_num / attention_normalizer  # [N, H, D]

#     # compute attention for visualization if needed
#     if output_attn:
#         attention=torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) #[N, N]
#         normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N,1]
#         attention=attention/normalizer


#     if output_attn:
#         return attn_output, attention
#     else:
#         return attn_output


# class TransConvLayer(nn.Module):
#     '''
#     transformer with fast attention
#     '''

#     def __init__(self, in_channels,
#                  out_channels,
#                  num_heads,
#                  use_weight=True):
#         super().__init__()
#         self.Wk = nn.Linear(in_channels, out_channels * num_heads)
#         self.Wq = nn.Linear(in_channels, out_channels * num_heads)
#         if use_weight:
#             self.Wv = nn.Linear(in_channels, out_channels * num_heads)

#         self.out_channels = out_channels
#         self.num_heads = num_heads
#         self.use_weight = use_weight

#     def reset_parameters(self):
#         self.Wk.reset_parameters()
#         self.Wq.reset_parameters()
#         if self.use_weight:
#             self.Wv.reset_parameters()

#     def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
#         # feature transformation
#         query = self.Wq(query_input).reshape(-1,
#                                              self.num_heads, self.out_channels)
#         key = self.Wk(source_input).reshape(-1,
#                                             self.num_heads, self.out_channels)
#         if self.use_weight:
#             value = self.Wv(source_input).reshape(-1,
#                                                   self.num_heads, self.out_channels)
#         else:
#             value = source_input.reshape(-1, 1, self.out_channels)

#         # compute full attentive aggregation
#         if output_attn:
#             attention_output, attn = full_attention_conv(
#                 query, key, value, output_attn)  # [N, H, D]
#         else:
#             attention_output = full_attention_conv(
#                 query, key, value)  # [N, H, D]

#         final_output = attention_output
#         final_output = final_output.mean(dim=1)

#         if output_attn:
#             return final_output, attn
#         else:
#             return final_output


# class TransConv(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
#                  alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False, jk=False):
#         super().__init__()
#         self.jk = jk
#         self.convs = nn.ModuleList()
#         self.fcs = nn.ModuleList()
#         self.fcs.append(nn.Linear(in_channels, hidden_channels))
#         self.bns = nn.ModuleList()
#         self.bns.append(nn.LayerNorm(hidden_channels))
#         for i in range(num_layers):
#             self.convs.append(
#                 TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
#             self.bns.append(nn.LayerNorm(hidden_channels))

#         self.dropout = dropout
#         self.activation = F.relu
#         self.use_bn = use_bn
#         self.residual = use_residual
#         self.alpha = alpha
#         self.use_act=use_act

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()
#         for fc in self.fcs:
#             fc.reset_parameters()

#     def forward(self, x, edge_index):
#         edge_weight = None
#         layer_ = []
#         x_local = 0
#         # input MLP layer
#         x = self.fcs[0](x)
#         if self.use_bn:
#             x = self.bns[0](x)
#         x = self.activation(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         layer_.append(x)

#         for i, conv in enumerate(self.convs):
#             # graph convolution with full attention aggregation
#             x = conv(x, x, edge_index, edge_weight)
#             if self.residual:
#                 x = self.alpha * x + (1-self.alpha) * layer_[i]
#             if self.use_bn:
#                 x = self.bns[i+1](x)
#             if self.use_act:
#                 x = self.activation(x) 
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             layer_.append(x)
#             if self.jk:
#                 x_local = x_local + x
#             else:
#                 x_local = x
#         x = x_local
#         return x

#     def get_attentions(self, x):
#         layer_, attentions = [], []
#         x = self.fcs[0](x)
#         if self.use_bn:
#             x = self.bns[0](x)
#         x = self.activation(x)
#         layer_.append(x)
#         for i, conv in enumerate(self.convs):
#             x, attn = conv(x, x, output_attn=True)
#             attentions.append(attn)
#             if self.residual:
#                 x = self.alpha * x + (1 - self.alpha) * layer_[i]
#             if self.use_bn:
#                 x = self.bns[i + 1](x)
#             layer_.append(x)
#         return torch.stack(attentions, dim=0)  # [layer num, N, N]

# class SGFormer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, 
#                  alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=False, graph_weight=0.8, gnn=None, aggregate='add', jk=False):
#         super().__init__()
#         self.trans_conv=TransConv(in_channels,hidden_channels,num_layers,num_heads,alpha,dropout,use_bn,use_residual,use_weight,jk=jk)
#         self.gnn=gnn
#         self.use_graph=use_graph
#         self.graph_weight=graph_weight
#         self.use_act=use_act
#         self.aggregate=aggregate

#         if aggregate=='add':
#             self.fc=nn.Linear(hidden_channels,out_channels)
#         elif aggregate=='cat':
#             self.fc=nn.Linear(2*hidden_channels,out_channels)
#         else:
#             raise ValueError(f'Invalid aggregate type:{aggregate}')
        
#         self.params1=list(self.trans_conv.parameters())
#         self.params2=list(self.gnn.parameters()) if self.gnn is not None else []
#         self.params2.extend(list(self.fc.parameters()) )

#     def forward(self,x,edge_index):
#         x1=self.trans_conv(x,edge_index)
#         if self.use_graph:
#             x2=self.gnn(x,edge_index)
#             if self.aggregate=='add':
#                 x=self.graph_weight*x2+(1-self.graph_weight)*x1
#             else:
#                 x=torch.cat((x1,x2),dim=1)
#         else:
#             x=x1
#         x=self.fc(x)
#         return x
    
#     def get_attentions(self, x):
#         attns=self.trans_conv.get_attentions(x) # [layer num, N, N]

#         return attns

#     def reset_parameters(self):
#         self.trans_conv.reset_parameters()
#         if self.use_graph:
#             self.gnn.reset_parameters()



# Save this file as sgformer_adapted.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

# ===============================================================================================
# HELPER CLASSES AND FUNCTIONS FOR SGFORMER (COPIED FROM ORIGINAL)
# ===============================================================================================

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, use_bn=True):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

def full_attention_conv(qs, ks, vs):
    qs = qs / torch.norm(qs, p=2)
    ks = ks / torch.norm(ks, p=2)
    N = qs.shape[0]
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)
    attention_num += N * vs
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer
    return attn_output

class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input):
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)
        attention_output = full_attention_conv(query, key, value)
        final_output = attention_output.mean(dim=1)
        return final_output

class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False, jk=False):
        super().__init__()
        self.jk = jk
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index=None):
        layer_ = []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)
        x_local = 0
        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
            if self.jk:
                x_local += x
            else:
                x_local = x
        return x_local

# ===============================================================================================
# INTERNAL SGFORMER CLASS (REFACTORED WITH get_embeddings)
# ===============================================================================================

class _SGFormer(nn.Module):
    def __init__(self, hidden_channels, out_channels, gnn, aggregate='add',
                 use_graph=True, graph_weight=0.8):
        super(_SGFormer, self).__init__()
        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')
        
        # Combine parameters for optimizer
        self.params1 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2 = list(self.fc.parameters())

    def reset_parameters(self):
        if self.use_graph and self.gnn is not None:
            self.gnn.reset_parameters()
        self.fc.reset_parameters()

    def get_embeddings(self, x_trans, x_feat, edge_index):
        """
        Generates the final node embeddings before the classification layer.
        """
        if self.use_graph and self.gnn is not None:
            x_gnn = self.gnn(x_feat, edge_index)
            if self.aggregate == 'add':
                final_emb = self.graph_weight * x_gnn + (1 - self.graph_weight) * x_trans
            else: # cat
                final_emb = torch.cat((x_trans, x_gnn), dim=1)
        else:
            final_emb = x_trans
        return final_emb

    def forward(self, x_trans, x_feat, edge_index):
        """
        The forward pass now uses get_embeddings.
        """
        embeddings = self.get_embeddings(x_trans, x_feat, edge_index)
        output = self.fc(embeddings)
        return output

# ===============================================================================================
# THE NEW WRAPPER CLASS (This is the one you should import and use)
# ===============================================================================================

class SGFormerAdapted(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 local_layers=2, dropout=0.5, heads=1,
                 # SGFormer specific args mapped from GNN args
                 res=True, ln=True, jk=False,
                 # Other args for compatibility
                 gnn=None, pre_ln=None, pre_linear=None, bn=None):
        
        super(SGFormerAdapted, self).__init__()

        # 1. The Transformer component of SGFormer
        self.trans_conv = TransConv(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=local_layers,
            num_heads=heads,
            dropout=dropout,
            use_bn=ln,
            use_residual=res,
            jk=jk
        )
        
        # 2. The GNN component of SGFormer
        # We instantiate the GCN part directly inside the wrapper.
        gnn_component = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels, # GNN part outputs hidden features
            num_layers=local_layers,
            dropout=dropout,
            use_bn=True # GCN part typically uses BatchNorm
        )

        # 3. The main _SGFormer model that combines them
        self.sgformer = _SGFormer(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            gnn=gnn_component,
            aggregate='add' # Using 'add' as a sensible default
        )
        
        # This is for the optimizer to get all parameters
        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.sgformer.parameters())

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        self.sgformer.reset_parameters()

    def get_embeddings(self, x, edge_index):
        # First, get the transformer part's output
        x_trans = self.trans_conv(x, edge_index)
        # Then, call the internal get_embeddings which combines it with the GNN part
        return self.sgformer.get_embeddings(x_trans, x, edge_index)

    def forward(self, x, edge_index):
        # The forward pass now mirrors the logic but is cleanly separated
        embeddings = self.get_embeddings(x, edge_index)
        output = self.sgformer.fc(embeddings) # Use the final fc layer from the internal model
        return output