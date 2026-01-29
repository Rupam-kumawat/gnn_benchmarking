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
from model import MPNNs
from dataset import load_dataset
from data_utils import load_fixed_splits
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

# =====================================================================================
#   Configuration
# =====================================================================================

K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5, 10]

# =====================================================================================
#   Ranking Metrics Helper
# =====================================================================================

def calculate_ranking_metrics(y_true, y_score, k_percentages):
    y_true = np.asarray(y_true).squeeze()
    y_score = np.asarray(y_score).squeeze()
    
    # Sort by score descending
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
        
        if total_positives > 0:
            metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
        else:
            metrics[f'recall_{metric_key}'] = 0.0
        
        # NDCG@K
        true_relevance = y_true.reshape(1, -1)
        pred_scores = y_score.reshape(1, -1)
        metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

    return metrics

# =====================================================================================
#   Arg Parsing Helper
# =====================================================================================

def parse_folder_to_args(param_string, default_args):
    """Parses folder string to reconstruct model arguments."""
    args = argparse.Namespace(**vars(default_args))
    parts = param_string.split('_')
    
    key_map = {
        'lr': 'lr', 'wd': 'weight_decay', 'do': 'dropout', 'hid': 'hidden_channels',
        'layers': 'local_layers', 'heads': 'num_heads', 'metric': 'metric'
    }
    
    if 'res' in parts: args.res = True
    if 'ln' in parts: args.ln = True
    if 'bn' in parts: args.bn = True
    if 'jk' in parts: args.jk = True
    if 'class_weight' in parts: args.use_class_weight = True

    for part in parts:
        if '-' in part:
            k, v = part.split('-', 1)
            if k in key_map:
                arg_name = key_map[k]
                if arg_name in ['local_layers', 'num_heads', 'hidden_channels']:
                    setattr(args, arg_name, int(v))
                elif arg_name in ['lr', 'weight_decay', 'dropout']:
                    setattr(args, arg_name, float(v))
                else:
                    setattr(args, arg_name, v)
    return args

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =====================================================================================
#   MAIN
# =====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='questions')
    parser.add_argument('--model_name', type=str, required=True, help='gcn, sage, gat')
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--base_results_dir', type=str, default='results_weighted')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--filter', type=str, default=None)
    
    # Model Defaults (Overwritten by folder parsing)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--local_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true')
    parser.add_argument('--ln', action='store_true')
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--jk', action='store_true')
    
    # Parser placeholders
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--metric', type=str, default='rocauc')
    parser.add_argument('--use_class_weight', action='store_true')
    parser.add_argument('--rand_split', action='store_true')
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)

    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    fix_seed(args.seed)

    # 1. Load Dataset
    print(f"--- Loading {args.dataset} Dataset ---")
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    
    # 2. Splits
    if args.rand_split:
        print("Using Random Splits")
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]
    else:
        print("Using Fixed Splits")
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)
    
    # Pre-process graph
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    print(f"Nodes: {n}, Classes: {c}, Features: {d}")

    # 3. Locate Experiments
    base_path = f'{args.base_results_dir}/{args.dataset}/{args.model_name}'
    if not os.path.exists(base_path):
        print(f"Error: Directory not found: {base_path}")
        return

    experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent', 'topk_results'] and not d.startswith('.')]
    if args.filter:
        experiment_dirs = [d for d in experiment_dirs if args.filter in d]

    # 4. Process
    for param_string in experiment_dirs:
        print(f"\nProcessing: {param_string}")
        
        # Prepare Output Directory inside experiment folder
        exp_path = os.path.join(base_path, param_string)
        out_dir = os.path.join(exp_path, 'topk_results')
        os.makedirs(out_dir, exist_ok=True)

        run_args = parse_folder_to_args(param_string, args)
        run_args.gnn = args.model_name

        all_ranking_metrics = {f'{m}_at_{k}_pct': [] for k in K_PERCENTAGES for m in ['precision', 'recall', 'ndcg']}

        for run in range(args.runs):
            run_dir = os.path.join(exp_path, f'run_{run}')
            model_path = os.path.join(run_dir, 'model.pt')
            
            if not os.path.exists(model_path): continue

            try:
                model = MPNNs(d, run_args.hidden_channels, c, 
                              local_layers=run_args.local_layers, 
                              dropout=run_args.dropout,
                              heads=run_args.num_heads, 
                              pre_ln=run_args.pre_ln, pre_linear=run_args.pre_linear,
                              res=run_args.res, ln=run_args.ln, bn=run_args.bn, jk=run_args.jk, 
                              gnn=run_args.gnn).to(device)
                
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                # Inference
                split_idx = split_idx_lst[run]
                test_idx = split_idx['test'].to(device)
                
                with torch.no_grad():
                    out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                    
                    if c == 2 or c == 1:
                        probs = torch.sigmoid(out[test_idx])
                        if probs.shape[1] == 1:
                            y_score = probs.cpu().numpy().squeeze()
                        else:
                            y_score = probs[:, 1].cpu().numpy()
                    else:
                        probs = torch.softmax(out[test_idx], dim=1)
                        y_score = probs[:, 1].cpu().numpy()

                    y_true = dataset.label[test_idx].cpu().numpy().squeeze()
                    
                    # Calculate Metrics
                    rank_metrics = calculate_ranking_metrics(y_true, y_score, K_PERCENTAGES)
                    for k, v in rank_metrics.items(): all_ranking_metrics[k].append(v)

                    # --- NON-INTRUSIVE SAVE ---
                    # 1. Read original metrics (to keep best_validation_metric)
                    orig_metrics_path = os.path.join(run_dir, 'metrics.json')
                    run_data = {}
                    if os.path.exists(orig_metrics_path):
                        with open(orig_metrics_path, 'r') as f:
                            run_data = json.load(f)
                    
                    # 2. Update and Save to NEW folder
                    run_data.update(rank_metrics)
                    new_metrics_path = os.path.join(out_dir, f'run_{run}_metrics.json')
                    
                    with open(new_metrics_path, 'w') as f:
                        json.dump(run_data, f, indent=4)

            except Exception as e:
                print(f"  Run {run} failed: {e}")
                continue

        # Save Summary to topk_results
        summary = {}
        for key, values in all_ranking_metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
        
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

    print("\nDone. Results saved in 'topk_results' folder.")

if __name__ == "__main__":
    main()