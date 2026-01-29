# parse_graphclass.py
import argparse

def parser_add_main_args(parser):
    # --- Dataset and Experiment Setup ---
    parser.add_argument('--dataset_type', type=str, default='tu', choices=['ogb', 'tu'], help='Type of dataset')
    parser.add_argument('--dataset', type=str, default='PROTEINS', help='Dataset name (e.g., ogbg-molhiv, DD, PROTEINS)')
    parser.add_argument('--device', type=int, default=0, help='Which GPU to use')
    parser.add_argument('--runs', type=int, default=3, help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # --- NEW: Metric for model selection ---
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'balacc', 'prauc'],
                        help='Metric to use for selecting the best model epoch')

    # --- Model Selection ---
    parser.add_argument('--model_family', type=str, default='gnn', choices=['gnn', 'gt'], help='Family of model to use')
    parser.add_argument('--model_name', type=str, default='gcn', 
                        choices=['gcn', 'gat', 'sage', 'gin', 'graphormer', 'graphit', 'gps', 'subgraphormer','graphvit'], 
                        help='Name of the model to use')
    
    # --- GNN & GT Hyperparameters ---
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN/Transformer layers')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Dimensionality of hidden units')
    parser.add_argument('--nhead', type=int, default=1, help='Number of attention heads (for GAT and Transformers)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--pool', type=str, default='mean', choices=['mean', 'add'], help='Graph pooling method')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 penalty)')
    
    # In parse_graphclass.py
    parser.add_argument('--use_bn', action='store_true', help='Enable batch norm in GNNs')
    parser.add_argument('--use_residual', action='store_true', help='Enable residual connections in GNNs')
    # In parse_graphclass.py, inside parser_add_main_args function

# --- GPS-specific Hyperparameters ---
    parser.add_argument('--local_gnn', type=str, default='GINE', choices=['GINE', 'GatedGCN'])
    parser.add_argument('--global_model', type=str, default='Transformer', choices=['Transformer'])

    # Positional Encoding Hyperparameters
    parser.add_argument('--use_lap_pe', action='store_true', help='Enable Laplacian Positional Encoding')
    parser.add_argument('--lap_pe_dim', type=int, default=16, help='Dimension of Laplacian PE')
    
    parser.add_argument('--subgraphormer_model_name', type=str, default='Subgraphormer_PE',
                        choices=['Subgraphormer', 'Subgraphormer_PE'],
                        help='Which Subgraphormer variant to use.')
    parser.add_argument('--subgraphormer_max_dis', type=int, default=5,
                        help='Maximum distance for node marking in Subgraphormer.')
    parser.add_argument('--subgraphormer_attention_type', type=str, default='Gat',
                        choices=['Gat', 'GatV2', 'Transformer_conv'],
                        help='Attention type for Subgraphormer.')
    parser.add_argument('--subgraphormer_aggs', type=str, default='uL,vL,vv',
                        help='Comma-separated list of aggregations for Subgraphormer (e.g., uL,vL,vv).')
    # parser.add_argument('--subgraphormer_sum_pooling', action='store_true',
    #                     help='Use sum pooling instead of mean pooling in Subgraphormer.')
    parser.add_argument('--subgraphormer_num_eigen_vectors', type=int, default=8,
                        help='Number of eigenvectors for positional encoding in Subgraphormer.')
    parser.add_argument('--subgraphormer_atom_encoder_linear', action='store_true',
                        help='Use a linear layer for atom encoding instead of an embedding table.')
    parser.add_argument('--subgraphormer_use_edge_attr', action='store_true',
                        help='Use edge attributes in Subgraphormer.')
    parser.add_argument('--subgraphormer_layer_encoder_linear', action='store_true',
                        help='Use a linear layer for the layer encoder instead of an MLP.')
    
    parser.add_argument('--graphvit_policy', type=str, default='node',
                        choices=['node', 'edge', 'rw', 'khop'],
                        help='Subgraph extraction policy for Graph-ViT.')
    parser.add_argument('--graphvit_k', type=int, default=16,
                        help='Number of subgraphs to extract for Graph-ViT.')
    parser.add_argument('--graphvit_max_nodes_subgraph', type=int, default=32,
                        help='Maximum number of nodes in each subgraph.')
    parser.add_argument('--graphvit_gnn_type', type=str, default='gin',
                        choices=['gin', 'gcn'],
                        help='GNN encoder type for Graph-ViT.')
    parser.add_argument('--graphvit_gnn_layers', type=int, default=2,
                        help='Number of layers for the GNN encoder in Graph-ViT.')