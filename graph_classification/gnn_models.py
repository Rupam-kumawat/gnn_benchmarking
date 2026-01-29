


# # # gnn_models.py

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_add_pool

# # # OGB-specific import for feature encoding
# # try:
# #     from ogb.graphproppred.mol_encoder import AtomEncoder
# # except ImportError:
# #     AtomEncoder = None

# # class GCN(torch.nn.Module):
# #     def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False):
# #         super().__init__()
# #         # --- MODIFICATION: Added Encoder ---
# #         if use_ogb_features and AtomEncoder is not None:
# #             self.encoder = AtomEncoder(hidden)
# #         else:
# #             self.encoder = nn.Linear(num_features, hidden)
        
# #         input_dim = hidden

# #         self.convs = torch.nn.ModuleList()
# #         self.convs.append(GCNConv(input_dim, hidden))
# #         for _ in range(num_layers - 2):
# #             self.convs.append(GCNConv(hidden, hidden))
# #         self.convs.append(GCNConv(hidden, hidden))
# #         self.lin = torch.nn.Linear(hidden, num_classes)
# #         self.dropout = dropout
# #         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         # The AtomEncoder does not have a reset_parameters method
# #         if not isinstance(self.encoder, AtomEncoder):
# #             self.encoder.reset_parameters()
# #         for conv in self.convs:
# #             conv.reset_parameters()
# #         self.lin.reset_parameters()

# #     def forward(self, data):
# #         x, edge_index, batch = data.x, data.edge_index, data.batch
# #         # --- MODIFICATION: Use Encoder ---
# #         x = self.encoder(x)
        
# #         for conv in self.convs:
# #             x = conv(x, edge_index).relu()
# #             x = F.dropout(x, p=self.dropout, training=self.training)
# #         x = self.pool(x, batch)
# #         return self.lin(x)

# # class GAT(torch.nn.Module):
# #     def __init__(self, num_features, num_classes, hidden=8, heads=8, num_layers=3, dropout=0.6, pool='mean', use_ogb_features=False):
# #         super().__init__()
# #         # --- MODIFICATION: Added Encoder ---
# #         if use_ogb_features and AtomEncoder is not None:
# #             self.encoder = AtomEncoder(hidden * heads) # GAT's first layer output is hidden * heads
# #             input_dim = hidden * heads
# #         else:
# #             self.encoder = nn.Linear(num_features, hidden * heads)
# #             input_dim = hidden * heads
            
# #         self.convs = torch.nn.ModuleList()
# #         self.convs.append(GATConv(input_dim, hidden, heads=heads, dropout=dropout))
# #         for _ in range(num_layers - 2):
# #             self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
# #         self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout))
# #         self.lin = torch.nn.Linear(hidden, num_classes)
# #         self.dropout = dropout
# #         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         if not isinstance(self.encoder, AtomEncoder):
# #             self.encoder.reset_parameters()
# #         for conv in self.convs:
# #             conv.reset_parameters()
# #         self.lin.reset_parameters()

# #     def forward(self, data):
# #         x, edge_index, batch = data.x, data.edge_index, data.batch
# #         # --- MODIFICATION: Use Encoder ---
# #         x = self.encoder(x)

# #         for i, conv in enumerate(self.convs):
# #             x = conv(x, edge_index)
# #             if i < len(self.convs) - 1:
# #                 x = F.elu(x)
# #             x = F.dropout(x, p=self.dropout, training=self.training)
# #         x = self.pool(x, batch)
# #         return self.lin(x)

# # class SAGE(torch.nn.Module):
# #     def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False):
# #         super().__init__()
# #         # --- MODIFICATION: Added Encoder ---
# #         if use_ogb_features and AtomEncoder is not None:
# #             self.encoder = AtomEncoder(hidden)
# #         else:
# #             self.encoder = nn.Linear(num_features, hidden)
        
# #         input_dim = hidden

# #         self.convs = torch.nn.ModuleList()
# #         self.convs.append(SAGEConv(input_dim, hidden))
# #         for _ in range(num_layers - 2):
# #             self.convs.append(SAGEConv(hidden, hidden))
# #         self.convs.append(SAGEConv(hidden, hidden))
# #         self.lin = torch.nn.Linear(hidden, num_classes)
# #         self.dropout = dropout
# #         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         if not isinstance(self.encoder, AtomEncoder):
# #             self.encoder.reset_parameters()
# #         for conv in self.convs:
# #             conv.reset_parameters()
# #         self.lin.reset_parameters()

# #     def forward(self, data):
# #         x, edge_index, batch = data.x, data.edge_index, data.batch
# #         # --- MODIFICATION: Use Encoder ---
# #         x = self.encoder(x)
        
# #         for conv in self.convs:
# #             x = conv(x, edge_index).relu()
# #             x = F.dropout(x, p=self.dropout, training=self.training)
# #         x = self.pool(x, batch)
# #         return self.lin(x)

# # class GIN(torch.nn.Module):
# #     def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False):
# #         super().__init__()
# #         # --- MODIFICATION: Added Encoder ---
# #         if use_ogb_features and AtomEncoder is not None:
# #             self.encoder = AtomEncoder(hidden)
# #             input_dim = hidden # GIN's first MLP will take the encoded dim
# #         else:
# #             self.encoder = nn.Linear(num_features, hidden)
# #             input_dim = hidden
            
# #         self.convs = torch.nn.ModuleList()
# #         self.bns = torch.nn.ModuleList()
        
# #         # --- MODIFICATION: First MLP takes encoded dimension ---
# #         mlp_initial = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
# #         self.convs.append(GINConv(mlp_initial, train_eps=True))
# #         self.bns.append(nn.BatchNorm1d(hidden))

# #         for _ in range(num_layers - 1):
# #             mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
# #             self.convs.append(GINConv(mlp, train_eps=True))
# #             self.bns.append(nn.BatchNorm1d(hidden))
            
# #         self.lin = torch.nn.Linear(hidden, num_classes)
# #         self.dropout = dropout
# #         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         if not isinstance(self.encoder, AtomEncoder):
# #             self.encoder.reset_parameters()
# #         for conv in self.convs:
# #             conv.reset_parameters()
# #         for bn in self.bns:
# #             bn.reset_parameters()
# #         self.lin.reset_parameters()

# #     def forward(self, data):
# #         x, edge_index, batch = data.x, data.edge_index, data.batch
# #         # --- MODIFICATION: Use Encoder ---
# #         x = self.encoder(x)

# #         for conv, bn in zip(self.convs, self.bns):
# #             x = conv(x, edge_index)
# #             x = bn(x).relu()
# #             x = F.dropout(x, p=self.dropout, training=self.training)
# #         x = self.pool(x, batch)
# #         return self.lin(x)





# # gnn_models.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_add_pool

# # OGB-specific import for feature encoding
# try:
#     from ogb.graphproppred.mol_encoder import AtomEncoder
# except ImportError:
#     AtomEncoder = None

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
#         super().__init__()
#         self.use_bn = use_bn
#         self.use_residual = use_residual
        
#         if use_ogb_features and AtomEncoder is not None:
#             self.encoder = AtomEncoder(hidden)
#         else:
#             self.encoder = nn.Linear(num_features, hidden)
        
#         input_dim = hidden

#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()
        
#         for i in range(num_layers):
#             in_channels = input_dim if i == 0 else hidden
#             self.convs.append(GCNConv(in_channels, hidden))
#             if self.use_bn:
#                 self.bns.append(nn.BatchNorm1d(hidden))
        
#         self.lin = torch.nn.Linear(hidden, num_classes)
#         self.dropout = dropout
#         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
#         self.reset_parameters()

#     def reset_parameters(self):
#         if not isinstance(self.encoder, AtomEncoder):
#             self.encoder.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         if self.use_bn:
#             for bn in self.bns:
#                 bn.reset_parameters()
#         self.lin.reset_parameters()

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.encoder(x)
        
#         for i, conv in enumerate(self.convs):
#             h = x
#             x = conv(x, edge_index)
#             if self.use_bn:
#                 x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             if self.use_residual and h.shape == x.shape:
#                 x = h + x # Residual connection
                
#         x = self.pool(x, batch)
#         return self.lin(x)


# class GAT(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden=8, heads=8, num_layers=3, dropout=0.6, pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
#         super().__init__()
#         self.use_bn = use_bn
#         self.use_residual = use_residual
        
#         if use_ogb_features and AtomEncoder is not None:
#             self.encoder = AtomEncoder(hidden * heads)
#             input_dim = hidden * heads
#         else:
#             self.encoder = nn.Linear(num_features, hidden * heads)
#             input_dim = hidden * heads
            
#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()
        
#         self.convs.append(GATConv(input_dim, hidden, heads=heads, dropout=dropout))
#         if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden * heads))

#         for _ in range(num_layers - 2):
#             self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
#             if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden * heads))
            
#         self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout))
#         if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden))

#         self.lin = torch.nn.Linear(hidden, num_classes)
#         self.dropout = dropout
#         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
#         self.reset_parameters()

#     def reset_parameters(self):
#         if not isinstance(self.encoder, AtomEncoder):
#             self.encoder.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         if self.use_bn:
#             for bn in self.bns:
#                 bn.reset_parameters()
#         self.lin.reset_parameters()

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.encoder(x)

#         for i, conv in enumerate(self.convs):
#             h = x
#             x = conv(x, edge_index)
#             if self.use_bn:
#                 x = self.bns[i](x)
#             if i < len(self.convs) - 1:
#                 x = F.elu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             if self.use_residual and h.shape == x.shape:
#                 x = h + x # Residual connection
                
#         x = self.pool(x, batch)
#         return self.lin(x)


# class SAGE(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
#         super().__init__()
#         self.use_bn = use_bn
#         self.use_residual = use_residual

#         if use_ogb_features and AtomEncoder is not None:
#             self.encoder = AtomEncoder(hidden)
#         else:
#             self.encoder = nn.Linear(num_features, hidden)
        
#         input_dim = hidden

#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()

#         for i in range(num_layers):
#             in_channels = input_dim if i == 0 else hidden
#             self.convs.append(SAGEConv(in_channels, hidden))
#             if self.use_bn:
#                 self.bns.append(nn.BatchNorm1d(hidden))

#         self.lin = torch.nn.Linear(hidden, num_classes)
#         self.dropout = dropout
#         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
#         self.reset_parameters()

#     def reset_parameters(self):
#         if not isinstance(self.encoder, AtomEncoder):
#             self.encoder.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         if self.use_bn:
#             for bn in self.bns:
#                 bn.reset_parameters()
#         self.lin.reset_parameters()

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.encoder(x)
        
#         for i, conv in enumerate(self.convs):
#             h = x
#             x = conv(x, edge_index)
#             if self.use_bn:
#                 x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             if self.use_residual and h.shape == x.shape:
#                 x = h + x # Residual connection
                
#         x = self.pool(x, batch)
#         return self.lin(x)

# class GIN(torch.nn.Module):
#     # GIN already includes batch norm in its standard formulation.
#     # Residual connections are less common due to the internal MLP structure.
#     def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False):
#         super().__init__()
#         if use_ogb_features and AtomEncoder is not None:
#             self.encoder = AtomEncoder(hidden)
#         else:
#             self.encoder = nn.Linear(num_features, hidden)
        
#         input_dim = hidden
            
#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()
        
#         mlp_initial = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
#         self.convs.append(GINConv(mlp_initial, train_eps=True))
#         self.bns.append(nn.BatchNorm1d(hidden))

#         for _ in range(num_layers - 1):
#             mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
#             self.convs.append(GINConv(mlp, train_eps=True))
#             self.bns.append(nn.BatchNorm1d(hidden))
            
#         self.lin = torch.nn.Linear(hidden, num_classes)
#         self.dropout = dropout
#         self.pool = global_mean_pool if pool == 'mean' else global_add_pool
#         self.reset_parameters()

#     def reset_parameters(self):
#         if not isinstance(self.encoder, AtomEncoder):
#             self.encoder.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()
#         self.lin.reset_parameters()

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.encoder(x)

#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             x = bn(x).relu()
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.pool(x, batch)
#         return self.lin(x)









# gnn_models.py
# Drop-in upgraded models: GCN, GAT, SAGE, GIN
# Internals upgraded to GraphGym-style blocks while preserving original class names,
# argument lists and external behavior. Respects use_bn and use_residual flags (option B).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_add_pool

# OGB-specific import for feature encoding
try:
    from ogb.graphproppred.mol_encoder import AtomEncoder
except ImportError:
    AtomEncoder = None


class GraphGymLayer(nn.Module):
    """
    GraphGym-style block wrapper around a single conv operator.

    Structure (when flags enabled):
      BN_pre (optional, dim = pre_bn_dim)
      conv(x, edge_index)
      BN_post (optional, dim = post_bn_dim)
      Activation
      Dropout
      Residual (optional, shape-guarded)
      FFN block (post_dim -> 2*post_dim -> post_dim)
      Residual (optional)
      BN_ffn (optional)

    Note: pre_bn_dim and post_bn_dim must be provided because some convs (e.g. GAT with multi-head)
    change dimensionality in-between.
    """

    def __init__(
        self,
        conv,
        pre_bn_dim,
        post_bn_dim,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        ffn=True,
    ):
        super().__init__()
        self.conv = conv
        self.pre_bn_dim = pre_bn_dim
        self.post_bn_dim = post_bn_dim
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.ffn = ffn

        # Pre and post batchnorm (optional)
        if self.use_bn:
            self.bn_pre = nn.BatchNorm1d(pre_bn_dim)
            self.bn_post = nn.BatchNorm1d(post_bn_dim)
        else:
            self.bn_pre = None
            self.bn_post = None

        # Feed-forward block params (apply on post-conv dimension)
        if self.ffn:
            if self.use_bn:
                self.ffn_norm1 = nn.BatchNorm1d(post_bn_dim)
                self.ffn_norm2 = nn.BatchNorm1d(post_bn_dim)
            else:
                self.ffn_norm1 = None
                self.ffn_norm2 = None
            self.ffn_linear1 = nn.Linear(post_bn_dim, 2 * post_bn_dim)
            self.ffn_linear2 = nn.Linear(2 * post_bn_dim, post_bn_dim)
            self.ffn_drop1 = nn.Dropout(dropout)
            self.ffn_drop2 = nn.Dropout(dropout)

        self.act = nn.ReLU()

    def reset_parameters(self):
        # conv may or may not implement reset_parameters
        try:
            self.conv.reset_parameters()
        except Exception:
            pass
        if self.use_bn:
            try:
                self.bn_pre.reset_parameters()
                self.bn_post.reset_parameters()
            except Exception:
                pass
        if self.ffn:
            try:
                self.ffn_linear1.reset_parameters()
                self.ffn_linear2.reset_parameters()
                if self.use_bn:
                    self.ffn_norm1.reset_parameters()
                    self.ffn_norm2.reset_parameters()
            except Exception:
                pass

    def forward(self, x, edge_index):
        # x: [N, C_in] where C_in should equal pre_bn_dim
        h_in = x  # for first residual (pre)
        if self.use_bn and (self.bn_pre is not None):
            x = self.bn_pre(x)

        # Message passing
        x = self.conv(x, edge_index)

        # Post-conv normalization (on post_bn_dim)
        if self.use_bn and (self.bn_post is not None):
            x = self.bn_post(x)

        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Residual 1: only if shapes match
        if self.use_residual and h_in.shape == x.shape:
            x = x + h_in

        # Feed-forward block (post-dim -> 2*post-dim -> post-dim)
        if self.ffn:
            h_ff = x
            if self.use_bn and (self.ffn_norm1 is not None):
                x = self.ffn_norm1(x)
            x = self.act(self.ffn_linear1(x))
            x = self.ffn_drop1(x)
            x = self.ffn_linear2(x)
            x = self.ffn_drop2(x)
            # Residual 2
            if self.use_residual and h_ff.shape == x.shape:
                x = x + h_ff
            if self.use_bn and (self.ffn_norm2 is not None):
                x = self.ffn_norm2(x)

        return x


# ---------------------------
# Drop-in classes: same names & args as original
# ---------------------------

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5,
                 pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_residual = use_residual

        # encoder: either OGB AtomEncoder or Linear to hidden
        if use_ogb_features and AtomEncoder is not None:
            self.encoder = AtomEncoder(hidden)
            enc_out_dim = hidden
        else:
            self.encoder = nn.Linear(num_features, hidden)
            enc_out_dim = hidden

        # build GraphGym-style layers (pre_bn_dim = enc_out_dim for first, then hidden)
        self.layers = torch.nn.ModuleList()
        in_dim = enc_out_dim
        for i in range(num_layers):
            # conv maps hidden->hidden (we ensure encoder projects to hidden)
            conv = GCNConv(hidden, hidden)
            pre_bn_dim = in_dim
            post_bn_dim = hidden
            layer = GraphGymLayer(
                conv=conv,
                pre_bn_dim=pre_bn_dim,
                post_bn_dim=post_bn_dim,
                dropout=dropout,
                use_bn=use_bn,
                use_residual=use_residual,
                ffn=True,
            )
            self.layers.append(layer)
            in_dim = post_bn_dim  # subsequent layer's pre_bn_dim

        self.lin = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        # reset encoder if possible
        if not isinstance(self.encoder, AtomEncoder):
            try:
                self.encoder.reset_parameters()
            except Exception:
                pass
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except Exception:
                pass
        try:
            self.lin.reset_parameters()
        except Exception:
            pass

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.pool(x, batch)
        return self.lin(x)


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden=8, heads=8, num_layers=3, dropout=0.6,
                 pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.heads = heads
        self.hidden = hidden
        self.dropout = dropout
        # Encoder: original code encoded to hidden * heads when using non-AtomEncoder
        if use_ogb_features and AtomEncoder is not None:
            # preserve previous behavior: encoder outputs hidden * heads
            self.encoder = AtomEncoder(hidden * heads)
            enc_out_dim = hidden * heads
        else:
            self.encoder = nn.Linear(num_features, hidden * heads)
            enc_out_dim = hidden * heads

        # Build conv layers matching the original intent:
        # first conv: in=enc_out_dim, out=hidden, heads=heads -> outputs hidden*heads
        # middle convs: in=hidden*heads, out=hidden, heads=heads -> outputs hidden*heads
        # final conv: in=hidden*heads, out=hidden, heads=1, concat=False -> outputs hidden
        self.layers = torch.nn.ModuleList()
        in_dim = enc_out_dim

        if num_layers == 1:
            # single-layer case: produce final-style conv (heads=1, concat=False)
            conv = GATConv(in_dim, hidden, heads=1, concat=False, dropout=dropout)
            pre_bn_dim = in_dim
            post_bn_dim = hidden
            self.layers.append(GraphGymLayer(conv, pre_bn_dim, post_bn_dim,
                                             dropout=dropout, use_bn=use_bn,
                                             use_residual=use_residual, ffn=True))
        else:
            # first layer
            conv0 = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
            pre_bn_dim = in_dim
            post_bn_dim = hidden * heads  # concat=True -> hidden*heads
            self.layers.append(GraphGymLayer(conv0, pre_bn_dim, post_bn_dim,
                                             dropout=dropout, use_bn=use_bn,
                                             use_residual=use_residual, ffn=True))
            in_dim = post_bn_dim

            # middle layers (if any)
            for _ in range(max(0, num_layers - 2)):
                conv_mid = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
                pre_bn_dim = in_dim
                post_bn_dim = hidden * heads
                self.layers.append(GraphGymLayer(conv_mid, pre_bn_dim, post_bn_dim,
                                                 dropout=dropout, use_bn=use_bn,
                                                 use_residual=use_residual, ffn=True))
                in_dim = post_bn_dim

            # final layer: heads=1, concat=False -> outputs hidden
            conv_last = GATConv(in_dim, hidden, heads=1, concat=False, dropout=dropout)
            pre_bn_dim = in_dim
            post_bn_dim = hidden
            self.layers.append(GraphGymLayer(conv_last, pre_bn_dim, post_bn_dim,
                                             dropout=dropout, use_bn=use_bn,
                                             use_residual=use_residual, ffn=True))

        self.lin = torch.nn.Linear(hidden, num_classes)
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.encoder, AtomEncoder):
            try:
                self.encoder.reset_parameters()
            except Exception:
                pass
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except Exception:
                pass
        try:
            self.lin.reset_parameters()
        except Exception:
            pass

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)
        return self.lin(x)


class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5,
                 pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_residual = use_residual

        if use_ogb_features and AtomEncoder is not None:
            self.encoder = AtomEncoder(hidden)
            enc_out_dim = hidden
        else:
            self.encoder = nn.Linear(num_features, hidden)
            enc_out_dim = hidden

        self.layers = torch.nn.ModuleList()
        in_dim = enc_out_dim
        for i in range(num_layers):
            conv = SAGEConv(in_dim if i == 0 else hidden, hidden)
            pre_bn_dim = in_dim
            post_bn_dim = hidden
            self.layers.append(GraphGymLayer(conv, pre_bn_dim, post_bn_dim,
                                             dropout=dropout, use_bn=use_bn,
                                             use_residual=use_residual, ffn=True))
            in_dim = post_bn_dim

        self.lin = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.encoder, AtomEncoder):
            try:
                self.encoder.reset_parameters()
            except Exception:
                pass
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except Exception:
                pass
        try:
            self.lin.reset_parameters()
        except Exception:
            pass

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)
        return self.lin(x)


class GIN(torch.nn.Module):
    # GIN already includes internal MLP. We wrap each GINConv in a GraphGymLayer.
    def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5,
                 pool='mean', use_ogb_features=False, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        if use_ogb_features and AtomEncoder is not None:
            self.encoder = AtomEncoder(hidden)
            enc_out_dim = hidden
        else:
            self.encoder = nn.Linear(num_features, hidden)
            enc_out_dim = hidden

        self.layers = torch.nn.ModuleList()
        in_dim = enc_out_dim

        def make_mlp(in_dim_local=hidden):
            return nn.Sequential(
                nn.Linear(in_dim_local, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )

        # first conv: allow input dim different from hidden (we will use GINConv which expects mlp)
        for i in range(num_layers):
            mlp = make_mlp()
            conv = GINConv(mlp, train_eps=True)
            pre_bn_dim = in_dim
            post_bn_dim = hidden
            self.layers.append(GraphGymLayer(conv, pre_bn_dim, post_bn_dim,
                                             dropout=dropout, use_bn=use_bn,
                                             use_residual=True, ffn=True))
            in_dim = post_bn_dim

        self.lin = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.encoder, AtomEncoder):
            try:
                self.encoder.reset_parameters()
            except Exception:
                pass
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except Exception:
                pass
        try:
            self.lin.reset_parameters()
        except Exception:
            pass

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)
        return self.lin(x)
