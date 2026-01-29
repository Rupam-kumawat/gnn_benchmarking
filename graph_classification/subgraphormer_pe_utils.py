

# subgraphormer_pe_utils.py

import torch
import numpy as np
import torch_geometric as pyg

def get_laplacian(data, norm_type='none'):
    """Calculates the graph Laplacian."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj = pyg.utils.to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze()
    deg = pyg.utils.degree(edge_index[0], num_nodes, dtype=torch.float)

    if norm_type == 'none':
        return torch.diag(deg) - adj
    else:
        raise NotImplementedError("Only 'none' Laplacian is implemented.")

def sign_fliper(tensor):
    """Applies random sign flips for symmetry breaking."""
    N, K = tensor.shape
    # ================================================================= #
    # THE FIX: Change (1, N) to (1, K) to match the number of eigenvectors #
    # ================================================================= #
    sign_tensor = (np.random.randint(0, 2, (1, K)) * 2 - 1)
    # ================================================================= #
    return tensor * sign_tensor

def pad_pe(pe_matrix, D):
    """Pads the PE matrix if it has fewer than D dimensions."""
    N, K = pe_matrix.shape
    if K >= D:
        return pe_matrix[:, :D]
    padding = torch.zeros((N, D - K))
    return torch.hstack((pe_matrix, padding))

def get_laplacian_pe_for_kron_graph(data, pos_enc_dim=16):
    """
    Calculates the Laplacian Positional Encodings for the graph product.
    """
    if data.num_nodes == 0:
        # Handle empty graphs
        return torch.zeros((0, pos_enc_dim))

    L = get_laplacian(data, 'none')
    
    # Ensure k is less than N for eigendecomposition
    k = min(data.num_nodes - 2, pos_enc_dim)
    if k < 0:
        return torch.zeros((data.num_nodes**2, pos_enc_dim))

    EigVal, EigVec = np.linalg.eigh(L.numpy())
    
    # Sort eigenvalues and eigenvectors
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    
    # Select top-k eigenvectors (and flip signs)
    EigVec_top_k = sign_fliper(EigVec[:, 1:k+1])

    # Approximate eigenvectors of the Kronecker product
    k_sqrt = int(np.ceil(np.sqrt(pos_enc_dim)))
    
    # Ensure we don't select more eigenvectors than available
    k_sqrt = min(k_sqrt, EigVec_top_k.shape[1])
    
    EigVec_kron = np.kron(EigVec_top_k[:, :k_sqrt], EigVec_top_k[:, :k_sqrt])
    
    PE = torch.from_numpy(EigVec_kron).float()
    return pad_pe(pe_matrix=PE, D=pos_enc_dim)