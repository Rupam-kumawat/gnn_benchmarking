import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
from sklearn.metrics import ndcg_score

# --- Import Project Modules ---
from sgformer_adapted import SGFormerAdapted
from nodeformer_adapted import NodeFormerAdapted
from dataset import load_dataset
from data_utils import load_fixed_splits
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

# =====================================================================================
#   Ranking Metrics Helper
# =====================================================================================

def calculate_ranking_metrics(y_true, y_score, k_percentages):
    y_true = np.asarray(y_true).squeeze()
    y_score = np.asarray(y_score).squeeze()
    
    sort_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sort_indices]
    
    total_positives = np.sum(y_true)
    total_samples = len(y_true_sorted)
    metrics = {}
    
    for k_pct in k_percentages:
        k = int(np.ceil(total_samples * (k_pct / 100.0)))
        if k > total_samples: k = total_samples 
        if k == 0: continue
            
        top_k_labels = y_true_sorted[:k]
        num_positives_in_top_k = np.sum(top_k_labels)
        
        metric_key = f"at_{k_pct}_pct"
        metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
        metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives if total_positives > 0 else 0.0
        
        true_relevance = y_true.reshape(1, -1)
        pred_scores = y_score.reshape(1, -1)
        metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

    return metrics

# =====================================================================================
#   Universal Parameter Parser
# =====================================================================================

def parse_folder_to_args(param_string, model_name, default_args):
    """Parses directory names based on the model's specific sweep naming convention."""
    args = argparse.Namespace(**vars(default_args))
    parts = param_string.split('_')
    
    # Common logic for the Bash Sweep format: dim_X_layers_Y_loss_Z...
    for i, part in enumerate(parts):
        if part == 'dim':
            args.hidden_channels = int(parts[i+1])
        elif part == 'layers':
            args.local_layers = int(parts[i+1])
        elif part == 'loss' and parts[i+1] == 'focal':
            args.use_focal = True
            
    return args

# =====================================================================================
#   Model Factory
# =====================================================================================

def get_model(model_name, in_channels, out_channels, args):
    if model_name.lower() == 'sgformer':
        return SGFormerAdapted(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            local_layers=args.local_layers,
            dropout=args.dropout,
            heads=args.num_heads,
            res=args.res,
            ln=args.ln,
            jk=args.jk
        )
    elif model_name.lower() == 'nodeformer':
        return NodeFormerAdapted(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            local_layers=args.local_layers,
            dropout=args.dropout,
            heads=args.num_heads,
            res=args.res,
            ln=args.ln,
            jk=args.jk
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# =====================================================================================
#   MAIN
# =====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='questions')
    parser.add_argument('--model_name', type=str, required=True, help='sgformer OR nodeformer')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs to process')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='.././data')
    parser.add_argument('--base_results_dir', type=str, default='results')
    
    # Architecture Defaults
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--res', action='store_true', default=True)
    parser.add_argument('--ln', action='store_true', default=True)
    parser.add_argument('--jk', action='store_true', default=False)

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1: dataset.label = dataset.label.unsqueeze(1)
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    
    # Pre-process Graph
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    # 2. Locate Experiment Folders
    base_path = f'{args.base_results_dir}/{args.dataset}/{args.model_name}'
    experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    experiment_dirs = [d for d in experiment_dirs if "dim_" in d] # Ensure it's a sweep folder

    for param_string in experiment_dirs:
        print(f"\n--- Processing {args.model_name.upper()}: {param_string} ---")
        exp_path = os.path.join(base_path, param_string)
        out_dir = os.path.join(exp_path, 'topk_results')
        os.makedirs(out_dir, exist_ok=True)

        run_args = parse_folder_to_args(param_string, args.model_name, args)
        all_ranking_metrics = {f'{m}_at_{k}_pct': [] for k in [0.5, 1, 2, 3, 4, 5, 10] for m in ['precision', 'recall', 'ndcg']}

        for r in range(args.runs):
            run_folder = f'run_{r + 1}'
            model_path = os.path.join(exp_path, run_folder, 'model.pt')
            if not os.path.exists(model_path): continue

            try:
                model = get_model(args.model_name, d, c, run_args).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                test_idx = split_idx_lst[r]['test'].to(device)
                with torch.no_grad():
                    out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                    probs = torch.sigmoid(out[test_idx])
                    y_score = probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy().squeeze()
                    y_true = dataset.label[test_idx].cpu().numpy().squeeze()
                    
                    metrics = calculate_ranking_metrics(y_true, y_score, [0.5, 1, 2, 3, 4, 5, 10])
                    for k, v in metrics.items(): all_ranking_metrics[k].append(v)

            except Exception as e:
                print(f" Error in {run_folder}: {e}")

        # Save Summary
        summary = {f'{k}_mean': np.mean(v) for k, v in all_ranking_metrics.items() if v}
        summary.update({f'{k}_std': np.std(v) for k, v in all_ranking_metrics.items() if v})
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

    print(f"\n✅ Finished evaluation for {args.model_name}")

if __name__ == "__main__":
    main()





# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import os
# import json
# import random
# from sklearn.metrics import ndcg_score

# # --- Import Project Modules ---
# from sgformer_adapted import SGFormerAdapted 
# from dataset import load_dataset
# from data_utils import load_fixed_splits
# from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

# # =====================================================================================
# #   Ranking Metrics Helper
# # =====================================================================================

# def calculate_ranking_metrics(y_true, y_score, k_percentages):
#     y_true = np.asarray(y_true).squeeze()
#     y_score = np.asarray(y_score).squeeze()
    
#     # Sort by score descending
#     sort_indices = np.argsort(y_score)[::-1]
#     y_true_sorted = y_true[sort_indices]
    
#     total_positives = np.sum(y_true)
#     total_samples = len(y_true_sorted)
#     metrics = {}
    
#     for k_pct in k_percentages:
#         k = int(np.ceil(total_samples * (k_pct / 100.0)))
#         if k > total_samples: k = total_samples 
#         if k == 0: continue
            
#         top_k_labels = y_true_sorted[:k]
#         num_positives_in_top_k = np.sum(top_k_labels)
        
#         metric_key = f"at_{k_pct}_pct"
#         metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
        
#         if total_positives > 0:
#             metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
#         else:
#             metrics[f'recall_{metric_key}'] = 0.0
        
#         true_relevance = y_true.reshape(1, -1)
#         pred_scores = y_score.reshape(1, -1)
#         metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

#     return metrics

# # =====================================================================================
# #   Arg Parsing Helper (Matches Bash Script Naming)
# # =====================================================================================

# def parse_bash_folder_to_args(param_string, default_args):
#     """
#     Parses directories like: dim_128_layers_3_loss_focal_gamma_2.0_metric_rocauc
#     """
#     args = argparse.Namespace(**vars(default_args))
#     parts = param_string.split('_')
    
#     # Simple iterator to find values after keywords
#     for i, part in enumerate(parts):
#         if part == 'dim':
#             args.hidden_channels = int(parts[i+1])
#         elif part == 'layers':
#             args.local_layers = int(parts[i+1])
#         elif part == 'metric':
#             args.metric = parts[i+1]
#         elif part == 'loss' and parts[i+1] == 'focal':
#             args.use_focal = True
            
#     return args

# def fix_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

# # =====================================================================================
# #   MAIN
# # =====================================================================================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='questions')
#     parser.add_argument('--model_name', type=str, default='sgformer')
#     parser.add_argument('--runs', type=int, default=10) # Default matches your bash script
#     parser.add_argument('--data_dir', type=str, default='.././data')
#     parser.add_argument('--base_results_dir', type=str, default='results')
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--filter', type=str, default=None)
    
#     # Defaults (In case parsing fails)
#     parser.add_argument('--hidden_channels', type=int, default=64)
#     parser.add_argument('--local_layers', type=int, default=1)
#     parser.add_argument('--num_heads', type=int, default=1)
#     parser.add_argument('--dropout', type=float, default=0.2)
#     parser.add_argument('--res', action='store_true', default=True) # Bash uses --res
#     parser.add_argument('--ln', action='store_true', default=True)  # Bash uses --ln
#     parser.add_argument('--jk', action='store_true')
#     parser.add_argument('--rand_split', action='store_true')

#     args = parser.parse_args()
#     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
#     fix_seed(args.seed)

#     # 1. Load Dataset
#     dataset = load_dataset(args.data_dir, args.dataset)
#     if len(dataset.label.shape) == 1:
#         dataset.label = dataset.label.unsqueeze(1)
    
#     # 2. Splits
#     if args.rand_split:
#         split_idx_lst = [dataset.get_idx_split(train_prop=0.6, valid_prop=0.2) for _ in range(args.runs)]
#     else:
#         split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

#     dataset.label = dataset.label.to(device)
#     n = dataset.graph['num_nodes']
#     c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
#     d = dataset.graph['node_feat'].shape[1]
    
#     dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
#     dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
#     dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

#     # 3. Locate Experiments (results/questions/sgformer)
#     base_path = f'{args.base_results_dir}/{args.dataset}/{args.model_name}'
#     if not os.path.exists(base_path):
#         print(f"Error: Directory not found: {base_path}")
#         return

#     experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
#     # Filter out summary folders
#     experiment_dirs = [d for d in experiment_dirs if "topk" not in d and not d.startswith('.')]
    
#     if args.filter:
#         experiment_dirs = [d for d in experiment_dirs if args.filter in d]

#     # 4. Process Loop
#     for param_string in experiment_dirs:
#         print(f"\nProcessing Experiment: {param_string}")
        
#         exp_path = os.path.join(base_path, param_string)
#         out_dir = os.path.join(exp_path, 'topk_results')
#         os.makedirs(out_dir, exist_ok=True)

#         run_args = parse_bash_folder_to_args(param_string, args)
        
#         k_percentages = [0.5, 1, 2, 3, 4, 5, 10]
#         all_ranking_metrics = {f'{m}_at_{k}_pct': [] for k in k_percentages for m in ['precision', 'recall', 'ndcg']}

#         for run_idx in range(args.runs):
#             # Bash script uses 1-indexed run folders (run_1, run_2, ...)
#             run_folder = f'run_{run_idx + 1}'
#             run_dir = os.path.join(exp_path, run_folder)
#             model_path = os.path.join(run_dir, 'model.pt')
            
#             if not os.path.exists(model_path): continue

#             try:
#                 model = SGFormerAdapted(
#                     in_channels=d,
#                     hidden_channels=run_args.hidden_channels,
#                     out_channels=c,
#                     local_layers=run_args.local_layers,
#                     dropout=run_args.dropout,
#                     heads=run_args.num_heads,
#                     res=run_args.res,
#                     ln=run_args.ln,
#                     jk=run_args.jk
#                 ).to(device)
                
#                 model.load_state_dict(torch.load(model_path, map_location=device))
#                 model.eval()

#                 split_idx = split_idx_lst[run_idx]
#                 test_idx = split_idx['test'].to(device)
                
#                 with torch.no_grad():
#                     out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                    
#                     # For questions (binary/sigmoid)
#                     probs = torch.sigmoid(out[test_idx])
#                     # If single output column, squeeze; if multi-class, take class 1
#                     y_score = probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy().squeeze()
#                     y_true = dataset.label[test_idx].cpu().numpy().squeeze()
                    
#                     rank_metrics = calculate_ranking_metrics(y_true, y_score, k_percentages)
#                     for k, v in rank_metrics.items(): all_ranking_metrics[k].append(v)

#                     # Update existing metrics.json if present
#                     orig_metrics_path = os.path.join(run_dir, 'metrics.json')
#                     run_data = {}
#                     if os.path.exists(orig_metrics_path):
#                         with open(orig_metrics_path, 'r') as f:
#                             run_data = json.load(f)
                    
#                     run_data.update(rank_metrics)
#                     new_metrics_path = os.path.join(out_dir, f'{run_folder}_metrics.json')
                    
#                     with open(new_metrics_path, 'w') as f:
#                         json.dump(run_data, f, indent=4)

#             except Exception as e:
#                 print(f"  {run_folder} failed: {e}")
#                 continue

#         # Save Experiment Summary
#         summary = {}
#         for key, values in all_ranking_metrics.items():
#             if values:
#                 summary[f'{key}_mean'] = np.mean(values)
#                 summary[f'{key}_std'] = np.std(values)
        
#         with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
#             json.dump(summary, f, indent=4)

#     print("\n✅ Ranking evaluation for Bash sweep finished.")

# if __name__ == "__main__":
#     main()