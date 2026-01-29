
# subgraphormer_model.py

import torch
import torch.nn as nn
from subgraphormer_layers import Atom, Bond, PE_layer, Attention_block, Pooling, MLP
import torch_scatter as pys
import torch.nn.functional as F
import torch_geometric

class Subgraphormer(nn.Module):
    def __init__(self, num_features, num_classes, args, dataset_type='tu'):
        super().__init__()
        self.num_layers = args.num_layers
        self.aggs = args.subgraphormer_aggs.split(',')
        self.use_residual = args.use_residual
        self.dropout = args.dropout
        # self.use_sum_pooling = args.subgraphormer_sum_pooling
        # self.use_sum_pooling = (args.pool == 'add')
        self.pool_strategy = args.pool 
        self.model_name = args.subgraphormer_model_name
        
        self.use_pe = "_PE" in self.model_name
        mpnn_input_dim = args.hidden_channels
        if self.use_pe:
            mpnn_input_dim += args.subgraphormer_num_eigen_vectors

        self.atom_encoder = Atom(
            dim=args.hidden_channels,
            max_dis=args.subgraphormer_max_dis,
            use_linear=args.subgraphormer_atom_encoder_linear,
            atom_dim=num_features,
            dataset_type=dataset_type
        )
        
        if self.use_pe:
            self.pe_layer = PE_layer(args.subgraphormer_num_eigen_vectors)

        self.MPNNs = nn.ModuleList()
        self.EDGE_ENCODERs = nn.ModuleList()
        self.POINT_ENCODERs = nn.ModuleList()
        self.EPSs = nn.ParameterList()
        self.CAT_ENCODERs = nn.ModuleList()
        self.BNs = nn.ModuleList()
        self.DROPs = nn.ModuleList()

        for i in range(self.num_layers):
            mpnn_layer = nn.ModuleDict()
            edge_encoder_layer = nn.ModuleDict()
            point_encoder_layer = nn.ModuleDict()
            eps_layer = nn.ParameterDict()

            current_input_dim = mpnn_input_dim if i == 0 else args.hidden_channels

            for agg in self.aggs:
                if "L" in agg or "G" in agg:
                    mpnn_layer[agg] = Attention_block(
                        d=current_input_dim,
                        H=args.nhead, 
                        d_output=args.hidden_channels,
                        edge_dim=args.hidden_channels,
                        type=args.subgraphormer_attention_type
                    )
                    if "L" in agg and args.subgraphormer_use_edge_attr:
                        edge_encoder_layer[agg] = Bond(args.hidden_channels)
                else:
                    point_encoder_layer[agg] = MLP(current_input_dim, args.hidden_channels)
                    eps_layer[agg] = nn.Parameter(torch.zeros(1))
            
            self.MPNNs.append(mpnn_layer)
            self.EDGE_ENCODERs.append(edge_encoder_layer)
            self.POINT_ENCODERs.append(point_encoder_layer)
            self.EPSs.append(eps_layer)
            self.CAT_ENCODERs.append(MLP(args.hidden_channels * len(self.aggs), args.hidden_channels))
            self.BNs.append(nn.BatchNorm1d(args.hidden_channels))
            if self.dropout > 0:
                self.DROPs.append(nn.Dropout(self.dropout))
        
        self.pooling = Pooling(args.hidden_channels, num_classes)

    def forward(self, batch):
        batch = self.atom_encoder(batch)
        if self.use_pe:
            batch = self.pe_layer(batch)
            
        for i in range(self.num_layers):
            all_aggs_out = []
            for agg in self.aggs:
                if "L" in agg or "G" in agg:
                    edge_attr = batch.get(f"attrs_{agg}", None)
                    edge_attr_encoded = None
                    # ================================================================= #
                    # THE FIX: Use 'in' to check for key existence in ModuleDict        #
                    # ================================================================= #
                    if edge_attr is not None and agg in self.EDGE_ENCODERs[i]:
                        edge_attr_encoded = self.EDGE_ENCODERs[i][agg](batch.x, edge_attr)
                    
                    agg_out = self.MPNNs[i][agg](batch.x, batch[f"index_{agg}"], edge_attr_encoded)
                else: # GIN-style aggregation
                    self_feat = (1 + self.EPSs[i][agg]) * batch.x
                    src_nodes = batch[f'index_{agg}'][1]
                    neighbor_feat = pys.scatter(batch.x[src_nodes], batch[f'index_{agg}'][0], dim=0, reduce='add')
                    agg_out = self.POINT_ENCODERs[i][agg](self_feat + neighbor_feat)
                all_aggs_out.append(agg_out)
            
            h = torch.cat(all_aggs_out, dim=1)
            h = self.CAT_ENCODERs[i](h)
            h = F.relu(self.BNs[i](h))
            
            if self.dropout > 0:
                h = self.DROPs[i](h)
                
            if self.use_residual:
                # Add residual connection to the correct feature tensor
                if h.size() == batch.x.size():
                    batch.x = batch.x + h
                else: # Handle dimension changes after first layer
                    # This case needs a projection if dims don't match, but for now we'll just update
                    batch.x = h 
            else:
                batch.x = h
            
            if self.pool_strategy == 'add':
                pooled_x = torch_geometric.nn.global_add_pool(batch.x, batch.batch)
            else: # Default to mean pooling
                pooled_x = torch_geometric.nn.global_mean_pool(batch.x, batch.batch)
            
            return self.pooling.predict(pooled_x)
        # return self.pooling(batch)