import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
import glob
import sys

# --- Matplotlib for Headless Plotting ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

# --- Sklearn Metrics ---
from sklearn.metrics import ndcg_score

# --- Import YOUR Project Modules ---
# These must be in the same directory
from model import MPNNs
from dataset import load_dataset
from data_utils import load_fixed_splits
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

# =====================================================================================
#   Ranking Metrics Calculation
# =====================================================================================

def calculate_ranking_metrics(y_true, y_score, k_percentages):
    """
    Calculates Precision, Recall, and NDCG at top K%.
    y_true: Binary labels (0 or 1)
    y_score: Probability scores for class 1
    """
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

        # Precision@K
        metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
        
        # Recall@K
        metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives if total_positives > 0 else 0.0
        
        # NDCG@K
        # ndcg_score expects shape (n_samples, n_labels) -> (1, n_nodes)
        true_relevance = y_true.reshape(1, -1)
        pred_scores = y_score.reshape(1, -1)
        metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

    return metrics

# =====================================================================================
#   Plotting Function
# =====================================================================================

def save_ranking_plots(summary_data, k_percentages, output_dir):
    metric_bases = ['precision', 'recall', 'ndcg']
    for metric_base in metric_bases:
        try:
            means = [summary_data.get(f'{metric_base}_at_{k}_pct_mean', np.nan) for k in k_percentages]
            stds = [summary_data.get(f'{metric_base}_at_{k}_pct_std', np.nan) for k in k_percentages]
            
            means = np.array(means)
            stds = np.array(stds)
            valid_indices = ~np.isnan(means)
            
            if not np.any(valid_indices): continue

            x_axis = np.array(k_percentages)[valid_indices]
            plot_means = means[valid_indices]
            plot_stds = stds[valid_indices]

            plt.figure(figsize=(10, 6))
            plt.plot(x_axis, plot_means, 'o-', label='Mean')
            plt.fill_between(x_axis, plot_means - plot_stds, plot_means + plot_stds,
                             color='blue', alpha=0.2, label='Mean ± 1 Std Dev')
            
            plt.title(f'{metric_base.capitalize()} vs. Top K%')
            plt.xlabel('Top K (%)')
            plt.ylabel(f'Mean {metric_base.capitalize()}')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plot_filename = os.path.join(output_dir, f'{metric_base}_vs_k_percent.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plot for {metric_base}. Error: {e}")

# =====================================================================================
#   PDF Report Generator
# =====================================================================================

def generate_comparison_pdf(base_path, output_pdf_name):
    print(f"\n{'='*50}\nGenerating Comparison PDF: {output_pdf_name}\n{'='*50}")

    try:
        experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
    except FileNotFoundError:
        print("Base path not found.")
        return

    # Group experiments by params (excluding metric)
    grouped_experiments = {}

    for folder_name in experiment_dirs:
        parts = folder_name.split('_')
        params = {}
        metric = None
        
        for part in parts:
            if '-' in part:
                k, v = part.split('-', 1)
                if k == 'metric':
                    metric = v
                else:
                    params[k] = v
        
        if metric is None: continue 
        
        config_key = tuple(sorted(params.items()))
        if config_key not in grouped_experiments:
            grouped_experiments[config_key] = {}
        grouped_experiments[config_key][metric] = os.path.join(base_path, folder_name)

    pdf_path = os.path.join(base_path, output_pdf_name)
    
    with PdfPages(pdf_path) as pdf:
        sorted_keys = sorted(grouped_experiments.keys())
        
        for config_key in sorted_keys:
            paths = grouped_experiments[config_key]
            path_prauc = paths.get('prauc')
            path_rocauc = paths.get('rocauc')
            
            if not path_prauc and not path_rocauc: continue

            config_name_str = " | ".join([f"{k}={v}" for k, v in config_key])
            print(f"Adding page for: {config_name_str}")

            fig, axes = plt.subplots(2, 3, figsize=(18, 11))
            
            def plot_row(row_idx, folder_path, metric_label):
                metrics = ['precision', 'recall', 'ndcg']
                if folder_path and os.path.exists(os.path.join(folder_path, 'topk_percent')):
                    img_dir = os.path.join(folder_path, 'topk_percent')
                    for col_idx, m in enumerate(metrics):
                        img_path = os.path.join(img_dir, f'{m}_vs_k_percent.png')
                        ax = axes[row_idx, col_idx]
                        ax.axis('off') 
                        if os.path.exists(img_path):
                            img = mpimg.imread(img_path)
                            ax.imshow(img)
                        else:
                            ax.text(0.5, 0.5, "Plot not found", ha='center', va='center')
                else:
                    for col_idx in range(3):
                        ax = axes[row_idx, col_idx]
                        ax.axis('off')
                        ax.text(0.5, 0.5, f"No data for {metric_label}", ha='center', va='center')

            plot_row(0, path_prauc, 'prauc')
            plot_row(1, path_rocauc, 'rocauc')

            plt.suptitle(f"Config: {config_name_str}", fontsize=15, fontweight='bold', y=0.98)
            
            fig.text(0.5, 0.94, "METRIC: PRAUC (Precision-Recall AUC)", 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='darkblue', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            fig.text(0.5, 0.49, "METRIC: ROCAUC (Receiver Operating Characteristic AUC)", 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='darkred',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93]) 
            plt.subplots_adjust(hspace=0.4) 
            pdf.savefig(fig)
            plt.close()

    print(f"PDF Generated: {pdf_path}")

# =====================================================================================
#   Parameter Parsing
# =====================================================================================

def parse_folder_to_args(param_string, default_args):
    """
    Parses the folder name strings generated by main_weighted.py back into args.
    Key mappings based on generate_param_string in main_weighted.py
    """
    args = argparse.Namespace(**vars(default_args))
    parts = param_string.split('_')
    
    # Mappings from folder string keys to Argparse variable names
    key_map = {
        'lr': 'lr', 'wd': 'weight_decay', 'do': 'dropout', 'hid': 'hidden_channels',
        'layers': 'local_layers', 'heads': 'num_heads', 'metric': 'metric'
    }
    
    # Boolean flags in folder name
    if 'res' in parts: args.res = True
    if 'ln' in parts: args.ln = True
    if 'bn' in parts: args.bn = True
    if 'jk' in parts: args.jk = True
    if 'class_weight' in parts: args.use_class_weight = True

    for part in parts:
        if '-' in part:
            k, v = part.split('-', 1)
            if k in key_map:
                # Convert types
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
    # Required to locate data and models
    parser.add_argument('--dataset', type=str, default='questions')
    parser.add_argument('--model_name', type=str, required=True, help='gcn, sage, gat')
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--base_results_dir', type=str, default='results_weighted')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--filter', type=str, default=None)
    
    # Default Args for Model Constructor (will be overwritten by folder parsing)
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
    
    # Just to satisfy parser if needed
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--metric', type=str, default='rocauc')
    parser.add_argument('--use_class_weight', action='store_true')

    # Data splitting args (from main_weighted.py defaults)
    parser.add_argument('--rand_split', action='store_true')
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)

    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    fix_seed(args.seed)

    print(f"--- Loading {args.dataset} Dataset ---")
    # 1. Load Dataset (Exact logic from main_weighted.py)
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    
    # 2. Generate/Load Splits (Exact logic from main_weighted.py)
    if args.rand_split:
        print("Using Random Splits (re-generating using seed)")
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]
    else:
        print("Using Fixed Splits")
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)
    
    # Pre-process graph (MPNNs usually expects specific edge structure)
    # Copied from main_weighted.py "else: # Standard PyG GNNs" block
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
    experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
    
    if args.filter:
        experiment_dirs = [d for d in experiment_dirs if args.filter in d]

    k_percentages = [1] + list(range(5, 101, 5))

    # 4. Evaluate Each Experiment
    for param_string in experiment_dirs:
        print(f"\nProcessing: {param_string}")
        run_args = parse_folder_to_args(param_string, args)
        
        # Override specific args
        run_args.gnn = args.model_name # Ensure 'gnn' attribute exists for MPNNs

        experiment_path = os.path.join(base_path, param_string)
        all_ranking_metrics = {f'{m}_at_{k}_pct': [] for k in k_percentages for m in ['precision', 'recall', 'ndcg']}

        for run in range(args.runs):
            # Check if run folder exists
            run_dir = os.path.join(experiment_path, f'run_{run}')
            model_path = os.path.join(run_dir, 'model.pt')
            
            if not os.path.exists(model_path):
                print(f"  Missing model for run {run}, skipping.")
                continue

            # Initialize Model (MPNNs class)
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
            except Exception as e:
                print(f"  Error loading model: {e}")
                continue

            # Inference
            split_idx = split_idx_lst[run]
            test_idx = split_idx['test'].to(device)
            
            with torch.no_grad():
                out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                
                # Handle Output - Questions is Binary/Multilabel
                # 'out' is likely logits
                if c == 2 or c == 1:
                    probs = torch.sigmoid(out[test_idx])
                    # If model output is [N, 1], squeeze it. If [N, 2], take col 1
                    if probs.shape[1] == 1:
                        y_score = probs.cpu().numpy().squeeze()
                    else:
                        y_score = probs[:, 1].cpu().numpy()
                else:
                    # Multiclass
                    probs = torch.softmax(out[test_idx], dim=1)
                    # For ranking in multiclass, we usually rank based on the probability of the *true* class
                    # But standard TopK for "node classification" usually implies "Anomaly Detection" style binary tasks
                    # Assuming Questions is Binary based on context. 
                    # If it's truly multiclass, simple TopK is ambiguous without a target class.
                    # We will take max prob as confidence? Or just skip?
                    # Fallback: Assume class 1 is the positive class we care about
                    y_score = probs[:, 1].cpu().numpy()

                y_true = dataset.label[test_idx].cpu().numpy().squeeze()
                
                # Calculate Ranking
                rank_metrics = calculate_ranking_metrics(y_true, y_score, k_percentages)
                
                for k, v in rank_metrics.items():
                    all_ranking_metrics[k].append(v)

        # Save Summary and Plots for this param configuration
        summary_dir = os.path.join(experiment_path, 'topk_percent')
        os.makedirs(summary_dir, exist_ok=True)
        
        summary = {}
        for key, values in all_ranking_metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
        
        with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        save_ranking_plots(summary, k_percentages, summary_dir)

    # 5. Generate PDF Report
    generate_comparison_pdf(base_path, f"{args.dataset}_{args.model_name}_comparison.pdf")

if __name__ == "__main__":
    main()