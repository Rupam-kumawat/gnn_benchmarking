# parse.py


from model import MPNNs

def parse_method(args, n, c, d, device):
    
    model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout, 
    heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear, res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn = args.gnn).to(device)
    
    return model
        
# parse.py

# This function defines all command-line arguments for the main script.
def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'balacc', 'prauc', 'qwk', 'spearman', 'mae', 'rmse'],
                        help='evaluation metric')

    # GNN Architectures
    parser.add_argument('--gnn', type=str, default='gcn', help='GNN architecture to use',
                        choices=['gcn', 'gat', 'sage', 'gin', 'fsgcn','glognn','gprgnn','mlp'])
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
    # FSGCN-specific arguments
    parser.add_argument('--fsgcn_num_layers', type=int, default=3, 
                        help='Number of hops for FSGCN feature pre-computation.')
    parser.add_argument('--fsgcn_feat_type', type=str, default='all', 
                        choices=['all', 'homophily', 'heterophily'], 
                        help='Type of pre-computed features for FSGCN.')
    parser.add_argument('--fsgcn_layer_norm', action='store_true', 
                        help='Use layer norm in the FSGNN model.')
    
    # Training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Display and Utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    
    #glognn specific arguments
    parser.add_argument('--glognn_alpha', type=float, default=0.5, help='Alpha for GloGNN')
    parser.add_argument('--glognn_beta', type=float, default=1.0, help='Beta for GloGNN')
    parser.add_argument('--glognn_gamma', type=float, default=1.0, help='Gamma for GloGNN')
    parser.add_argument('--glognn_delta', type=float, default=0.5, help='Delta for GloGNN')
    parser.add_argument('--glognn_norm_func_id', type=int, default=1, choices=[1, 2], help='Norm function ID for GloGNN')
    parser.add_argument('--glognn_norm_layers', type=int, default=1, help='Number of norm layers for GloGNN')
    parser.add_argument('--glognn_orders', type=int, default=2, help='Orders for GloGNN')
    parser.add_argument('--glognn_orders_func_id', type=int, default=2, choices=[1, 2, 3], help='Orders function ID for GloGNN')
    
    #gprgnn specific arguments
    parser.add_argument('--gprgnn_ppnp', type=str, default='GPR_prop', choices=['PPNP', 'GPR_prop'],
                        help='Propagation type for GPRGNN')
    parser.add_argument('--gprgnn_K', type=int, default=10,
                        help='Number of propagation iterations for GPRGNN')
    parser.add_argument('--gprgnn_alpha', type=float, default=0.1,
                        help='Teleport probability alpha for GPRGNN')
    parser.add_argument('--gprgnn_dprate', type=float, default=0.5,
                        help='Dropout rate for GPRGNN propagation')
    parser.add_argument('--gprgnn_Init', type=str, default='PPR',
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS'],
                        help='Initialization for GPRGNN propagation weights')

    # ... (rest of the arguments remain) ...
    
# def parser_add_main_args(parser):
#     # dataset and evaluation
#     parser.add_argument('--dataset', type=str, default='roman-empire')
#     parser.add_argument('--data_dir', type=str, default='./data/')
#     parser.add_argument('--device', type=int, default=0,
#                         help='which gpu to use if any (default: 0)')
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--cpu', action='store_true')
#     parser.add_argument('--epochs', type=int, default=500)
#     parser.add_argument('--runs', type=int, default=1,
#                         help='number of distinct runs')
#     parser.add_argument('--train_prop', type=float, default=.5,
#                         help='training label proportion')
#     parser.add_argument('--valid_prop', type=float, default=.25,
#                         help='validation label proportion')
#     parser.add_argument('--rand_split', action='store_true',
#                         help='use random splits')
#     parser.add_argument('--rand_split_class', action='store_true',
#                         help='use random splits with a fixed number of labeled nodes for each class')
    
#     parser.add_argument('--label_num_per_class', type=int, default=20,
#                         help='labeled nodes per class(randomly selected)')
#     parser.add_argument('--valid_num', type=int, default=500,
#                         help='Total number of validation')
#     parser.add_argument('--test_num', type=int, default=1000,
#                         help='Total number of test')
    
#     parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
#                         help='evaluation metric')
#     parser.add_argument('--model', type=str, default='MPNN')
#     # GNN
#     parser.add_argument('--gnn', type=str, default='gcn')
#     parser.add_argument('--hidden_channels', type=int, default=256)
#     parser.add_argument('--local_layers', type=int, default=7)
#     parser.add_argument('--num_heads', type=int, default=1,
#                         help='number of heads for attention')
#     parser.add_argument('--pre_ln', action='store_true')
#     parser.add_argument('--pre_linear', action='store_true')
#     parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
#     parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
#     parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
#     parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
#     # training
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--weight_decay', type=float, default=5e-4)
#     parser.add_argument('--dropout', type=float, default=0.5)
#     # display and utility
#     parser.add_argument('--display_step', type=int,
#                         default=100, help='how often to print')
#     parser.add_argument('--save_model', action='store_true', help='whether to save model')
#     parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')



