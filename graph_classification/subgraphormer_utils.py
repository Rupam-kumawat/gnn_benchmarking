
# subgraphormer_utils.py

import torch
import torch_geometric as pyg
import torch_geometric.data as data
# Import from both preprocessing files
import subgraphormer_pre_processing as spp
import subgraphormer_pe_utils as pe_utils

def safe_subgraphormer_transform(args):
    base_transform = get_subgraphormer_transform(args)
    def transform(data):
        if getattr(data, "x", None) is None:
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
        return base_transform(data)
    return transform

def get_subgraphormer_transform(args):
    """
    Creates the data transform required by Subgraphormer.
    
    """
    use_sum_pooling = (args.pool == 'add')
    return subgraph_construction(args.subgraphormer_aggs.split(','), use_sum_pooling, args.subgraphormer_num_eigen_vectors)

# Add num_eigen_vectors to the function signature
def subgraph_construction(aggs, sum_pooling, num_eigen_vectors):
    def call(graph):
        # --- Start of existing code ---
        subgraph_node_indices = torch.arange(graph.num_nodes**2).view((graph.num_nodes, graph.num_nodes))
        adj_original = pyg.utils.to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze(0)
        apsp = spp.get_all_pairs_shortest_paths(adj=adj_original)
        apsp[torch.isinf(apsp)] = 1001.0
        apsp = apsp.to(int).flatten()

        subgraphs_x = spp.get_subgraph_node_features(original_graph_features=graph.x)
        
        data_dict = {
            "num_subgraphs_bch": graph.num_nodes,
            "orig_edge_index": graph.edge_index,
            "y": graph.y,
            "x": subgraphs_x,
            "d": apsp
        }

        if 'uv' in aggs: data_dict['index_uv'] = spp.get_edge_index_uv(subgraph_node_indices)
        if 'vu' in aggs: data_dict['index_vu'] = spp.get_edge_index_vu(subgraph_node_indices)
        if 'uu' in aggs: data_dict['index_uu'] = spp.get_edge_index_uu(subgraph_node_indices)
        if 'vv' in aggs: data_dict['index_vv'] = spp.get_edge_index_vv(subgraph_node_indices)
        if 'uL' in aggs:
            edge_index_uL = spp.get_edge_index_uL(subgraph_node_indices, graph.edge_index)
            data_dict['index_uL'] = edge_index_uL
            if graph.edge_attr is not None:
                data_dict['attrs_uL'] = spp.get_edge_attr_uL(subgraph_node_indices, graph.edge_attr)
        if 'vL' in aggs:
            data_dict['index_vL'] = spp.get_edge_index_vL(subgraph_node_indices, graph.edge_index)
            if graph.edge_attr is not None:
                data_dict['attrs_vL'] = spp.get_edge_attr_vL(subgraph_node_indices, graph.edge_attr)
        if sum_pooling:
            data_dict['index_uG'] = spp.get_edge_index_uG(subgraph_node_indices)
        else:
            data_dict['index_uG_pool'] = spp.get_edge_index_uG_efficient_pooling(graph.num_nodes)
        # --- End of existing code ---

        # ================================================================= #
        # THE FIX: Add the PE calculation to the data dictionary            #
        # ================================================================= #
        pe = pe_utils.get_laplacian_pe_for_kron_graph(data=graph, pos_enc_dim=num_eigen_vectors)
        data_dict['subgraph_PE'] = pe
        # ================================================================= #

        return data.Data(**data_dict)
    return call