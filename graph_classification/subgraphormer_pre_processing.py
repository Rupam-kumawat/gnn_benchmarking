# subgraphormer_pre_processing.py
import torch
import torch_scatter as pys

def get_all_pairs_shortest_paths(adj):
    spd = torch.where(~torch.eye(len(adj), dtype=bool) & (adj == 0),
                      torch.full_like(adj, float("inf")), adj)
    for k in range(len(spd)):
        dist_from_source_to_k = spd[:, [k]]
        dist_from_k_to_target = spd[[k], :]
        dist_from_source_to_target_via_k = dist_from_source_to_k + dist_from_k_to_target
        spd = torch.minimum(spd, dist_from_source_to_target_via_k)
    return spd

def get_subgraph_node_features(original_graph_features):
    N = original_graph_features.size(0)
    return original_graph_features.repeat(N, 1)

def get_edge_index_uv(subgraph_node_indices):
    src_nodes = subgraph_node_indices
    target_nodes = subgraph_node_indices
    return torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)

def get_edge_index_vu(subgraph_node_indices):
    src_nodes = subgraph_node_indices.T
    target_nodes = subgraph_node_indices
    return torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)

def get_edge_index_uu(subgraph_node_indices):
    target_nodes = subgraph_node_indices
    _, src_nodes = torch.stack(torch.broadcast_tensors(
        target_nodes, torch.diag(subgraph_node_indices)[:, None]))
    return torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)

def get_edge_index_vv(subgraph_node_indices):
    target_nodes = subgraph_node_indices
    _, src_nodes = torch.stack(torch.broadcast_tensors(
        target_nodes, torch.diag(subgraph_node_indices)[None, :]))
    return torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)

def get_edge_index_uL(subgraph_node_indices, edge_index_of_original_graph):
    adjusted_node_indices = subgraph_node_indices[None, None, :, 0]
    combined_indices = adjusted_node_indices + edge_index_of_original_graph[:, :, None]
    return combined_indices.flatten(start_dim=1)

def get_edge_attr_uL(subgraph_node_indices, edge_attr_of_original_graph):
    if len(edge_attr_of_original_graph.shape) == 1:
        edge_attr_of_original_graph = edge_attr_of_original_graph.unsqueeze(1)
    
    expanded_dim = edge_attr_of_original_graph[None, :, :]
    expanded_tensor = expanded_dim.expand(len(subgraph_node_indices), -1, -1)
    permuted_tensor = expanded_tensor.permute(2, 1, 0)
    return permuted_tensor.flatten(start_dim=1).T

def get_edge_index_vL(subgraph_node_indices, edge_index_of_original_graph):
    """Computes the edge index for the 'vL' aggregation."""
    num_nodes_per_subgraph = len(subgraph_node_indices)
    adjusted_edge_index = edge_index_of_original_graph[:, :, None] * num_nodes_per_subgraph
    indexed_subgraph = subgraph_node_indices[None, None, 0, :] + adjusted_edge_index
    return indexed_subgraph.flatten(start_dim=1)

def get_edge_attr_vL(subgraph_node_indices, edge_attr_of_original_graph):
    """Computes the edge attributes for the 'vL' aggregation."""
    # This is often the same as the uL attribute calculation
    return get_edge_attr_uL(subgraph_node_indices, edge_attr_of_original_graph)

def get_edge_index_uG(subgraph_node_indices):
    target_nodes, src_nodes = torch.broadcast_tensors(
        subgraph_node_indices[:, :, None], subgraph_node_indices[:, None, :])
    return torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)

def get_edge_index_uG_efficient_pooling(num_subgraphs):
    src_nodes = torch.arange(num_subgraphs**2)
    target_nodes = torch.repeat_interleave(torch.arange(num_subgraphs), num_subgraphs)
    return torch.stack((target_nodes, src_nodes))