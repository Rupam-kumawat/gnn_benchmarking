import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
import glob

# --- ADDED: Matplotlib and PDF backends ---
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

# --- Import all required metrics ---
from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score, ndcg_score

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from gnn_models import GCN, GAT, SAGE, GIN
from training_utils import evaluate_graphclass
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


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
        
        if total_positives > 0:
            metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
        else:
            metrics[f'recall_{metric_key}'] = 0.0
        
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
#   UPDATED FUNCTION: Generate Comparison PDF
# =====================================================================================

def generate_comparison_pdf(base_path, output_pdf_name="comparison_report.pdf"):
    """
    Scans the results folder, pairs experiments that differ only by metric (prauc vs rocauc),
    and generates a PDF comparing their plots with explicit headers.
    """
    print(f"\n{'='*50}\nGenerating Comparison PDF: {output_pdf_name}\n{'='*50}")

    try:
        experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
    except FileNotFoundError:
        print("Base path not found, skipping PDF generation.")
        return

    # Group experiments by everything EXCEPT the metric
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
        
        # Sort params to create a consistent key
        config_key = tuple(sorted(params.items()))
        
        if config_key not in grouped_experiments:
            grouped_experiments[config_key] = {}
        
        grouped_experiments[config_key][metric] = os.path.join(base_path, folder_name)

    pdf_path = os.path.join(base_path, output_pdf_name)
    
    with PdfPages(pdf_path) as pdf:
        count = 0
        sorted_keys = sorted(grouped_experiments.keys())
        
        for config_key in sorted_keys:
            paths = grouped_experiments[config_key]
            path_prauc = paths.get('prauc')
            path_rocauc = paths.get('rocauc')
            
            # Skip if we don't have at least one of them
            if not path_prauc and not path_rocauc:
                continue

            count += 1
            # Reconstruct a readable string for the "Configuration Name"
            config_name_str = " | ".join([f"{k}={v}" for k, v in config_key])
            print(f"Creating page for: {config_name_str}")

            # Create Figure: 2 Rows (Top=PRAUC, Bottom=ROCAUC), 3 Cols
            # Adjusted figsize to make room for headers
            fig, axes = plt.subplots(2, 3, figsize=(18, 11))
            
            # --- Helper to plot a row ---
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
                            # Only set column titles on the very top row if you want, 
                            # but here we rely on the row headers for context.
                        else:
                            ax.text(0.5, 0.5, "Plot not found", ha='center', va='center')
                else:
                    for col_idx in range(3):
                        ax = axes[row_idx, col_idx]
                        ax.axis('off')
                        ax.text(0.5, 0.5, f"No folder/data for {metric_label}", ha='center', va='center')

            # --- Fill Rows ---
            # ROW 0 (Top) = PRAUC
            plot_row(0, path_prauc, 'prauc')
            
            # ROW 1 (Bottom) = ROCAUC
            plot_row(1, path_rocauc, 'rocauc')

            # --- TEXT / LAYOUT CONFIGURATION ---
            
            # 1. Main Page Title (Configuration Name)
            plt.suptitle(f"Config: {config_name_str}", fontsize=15, fontweight='bold', y=0.98)
            
            # 2. Section Header for TOP Row (PRAUC)
            # Positioned at y=0.94 (just below title, above plots)
            fig.text(0.5, 0.94, "METRIC: PRAUC (Precision-Recall AUC)", 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='darkblue', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 3. Section Header for BOTTOM Row (ROCAUC)
            # Positioned at y=0.49 (middle of page)
            fig.text(0.5, 0.49, "METRIC: ROCAUC (Receiver Operating Characteristic AUC)", 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='darkred',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # Adjust layout to make space for the text headers
            # rect=[left, bottom, right, top]
            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93]) 
            
            # Extra adjustment to separate the top and bottom rows visually
            plt.subplots_adjust(hspace=0.4) 

            pdf.savefig(fig)
            plt.close()

    print(f"PDF Report generated successfully: {pdf_path}")


# =====================================================================================
#   Argument Parsing & Setup
# =====================================================================================

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_param_string_to_args(param_string, base_args):
    args = argparse.Namespace(**vars(base_args))
    parts = param_string.split('_')
    
    key_map = {
        'lr': 'lr', 'pool': 'pool', 'metric': 'metric', 'hid': 'hidden_channels',
        'do': 'dropout', 'layers': 'num_layers', 'heads': 'nhead', 'bn': 'use_bn',
        'res': 'use_residual', 'focal_g': 'focal_gamma', 'focal_a': 'focal_alpha'
    }
    type_map = {
        'lr': float, 'pool': str, 'metric': str, 'hid': int, 'do': float,
        'layers': int, 'heads': int, 'bn': lambda x: x.lower() == 'true',
        'res': lambda x: x.lower() == 'true', 'focal_g': float, 'focal_a': float
    }
    
    if 'class_weight' in parts: args.use_class_weight = True
    if any(p.startswith('focal_g-') for p in parts): args.use_focal_loss = True

    for part in parts:
        if '-' not in part: continue
        try:
            key_short, val = part.split('-', 1)
            if key_short in key_map:
                setattr(args, key_map[key_short], type_map[key_short](val))
        except Exception: pass
            
    return args

# =====================================================================================
#   MAIN
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Pipeline + PDF Reporting')
    
    # Required
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)

    # Defaults
    parser.add_argument('--model_family', type=str, default='gnn')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metric', type=str, default='rocauc')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--use_residual', action='store_true', default=False)
    parser.add_argument('--use_class_weight', action='store_true', default=False)
    parser.add_argument('--use_focal_loss', action='store_true', default=False)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    
    parser.add_argument('--filter', type=str, default=None, help='Filter specific folders')

    base_args = parser.parse_args()
    base_args.dataset_type = 'ogb' if 'ogb' in base_args.dataset else 'tu'
    device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(base_args.seed)
    
    # --- 1. Load Data ---
    print(f"--- Loading dataset {base_args.dataset} ---")
    if base_args.dataset_type == 'ogb':
        # with torch.serialization.safe_globals([DataEdgeAttr]):
        dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
        split_idx = dataset.get_idx_split()
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=base_args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=base_args.batch_size, shuffle=False)
        evaluator = Evaluator(name=base_args.dataset)
        num_tasks = dataset.num_tasks
        num_features = dataset.num_features
        is_binary = True 
    else:
        dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
        if dataset.num_node_features == 0:
            max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset if g.edge_index.numel() > 0)
            dataset.transform = OneHotDegree(max_degree=max_deg)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        test_size = len(dataset) // 10
        valid_size = len(dataset) // 10
        test_dataset = dataset[indices[:test_size]]
        valid_dataset = dataset[indices[test_size:test_size + valid_size]]
        valid_loader = DataLoader(valid_dataset, batch_size=base_args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=base_args.batch_size, shuffle=False)
        evaluator = None
        num_tasks = dataset.num_classes
        num_features = dataset.num_features
        is_binary = dataset.num_classes == 2

    k_percentages = [1] + list(range(5, 101, 5))

    # --- 2. Evaluate Experiments ---
    base_path = f'results_weighted/{base_args.dataset}/{base_args.model_name}'
    
    try:
        experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
        if base_args.filter:
            experiment_dirs = [d for d in experiment_dirs if base_args.filter in d]
    except FileNotFoundError:
        print(f"Base directory not found: {base_path}")
        return

    # Process each folder
    for param_string in experiment_dirs:
        experiment_path = os.path.join(base_path, param_string)
        
        # Check if summary already exists to save time? (Optional, currently runs fully)
        # if os.path.exists(os.path.join(experiment_path, 'topk_percent', 'summary.json')): continue

        print(f"\nEvaluating: {param_string}")
        try:
            args = parse_param_string_to_args(param_string, base_args)
        except:
            continue

        all_ranking_metrics = {f'{m}_at_{k}_pct': [] for k in k_percentages for m in ['precision', 'recall', 'ndcg']}

        for run in range(1, args.runs + 1):
            # Model Setup
            models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE}
            ModelClass = models[args.model_name]
            use_ogb = True if args.dataset_type == 'ogb' else False
            
            gnn_args = {
                'num_features': num_features, 'num_classes': num_tasks, 'hidden': args.hidden_channels,
                'num_layers': args.num_layers, 'dropout': args.dropout, 'pool': args.pool,
                'use_ogb_features': use_ogb, 'use_bn': args.use_bn, 'use_residual': args.use_residual
            }
            # if args.model_name == 'gat': gnn_args['num_heads'] = args.nhead
            
            # Load Model
            try:
                model = ModelClass(**gnn_args)
                model_path = os.path.join(experiment_path, f'run_{run}', 'model.pt')
                if not os.path.exists(model_path): continue
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
            except Exception as e:
                print(f"Error loading model run {run}: {e}")
                continue

            # Evaluate
            _, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
            y_test_true, y_test_pred, _ = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

            if is_binary:
                # Get best threshold from Val
                valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy() if y_valid_pred.shape[1] == 1 else torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
                valid_true = _ if _ is not None else np.zeros_like(valid_probs) 
                
                # Test Metrics
                test_probs_np = torch.sigmoid(y_test_pred).squeeze().cpu().numpy() if y_test_pred.shape[1] == 1 else torch.softmax(y_test_pred, dim=1)[:, 1].cpu().numpy()
                y_test_true_np = y_test_true.cpu().numpy().squeeze()
                
                ranking_metrics_run = calculate_ranking_metrics(y_test_true_np, test_probs_np, k_percentages)
                
                for k, v in ranking_metrics_run.items():
                    if k in all_ranking_metrics: all_ranking_metrics[k].append(v)

        # Save Summary & Plots
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

    # --- 3. NEW: Generate Comparison PDF ---
    generate_comparison_pdf(base_path, output_pdf_name=f"{base_args.dataset}_{base_args.model_name}_comparison.pdf")

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
# import glob

# # --- ADDED: Matplotlib for plotting ---
# import matplotlib
# matplotlib.use('Agg') # Use 'Agg' backend for headless servers
# import matplotlib.pyplot as plt

# # --- Import all required metrics ---
# from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score, ndcg_score

# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
# from torch_geometric.transforms import OneHotDegree
# from torch_geometric.utils import degree
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# # from parse_graphclass import parser_add_main_args # MODIFIED: We define args locally
# from gnn_models import GCN, GAT, SAGE, GIN
# # from ignore_gt_models import Graphormer, GraphiT, GPS
# from training_utils import evaluate_graphclass
# from torch_geometric.data.storage import GlobalStorage
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# # =====================================================================================
# #  MODIFIED: Ranking Metrics Helper (now uses percentages)
# # =====================================================================================

# def calculate_ranking_metrics(y_true, y_score, k_percentages):
#     """
#     Calculates Precision@K, Recall@K, and NDCG@K for a list of K percentages.
#     Assumes y_true are 0/1 relevance labels and y_score are prediction scores.
#     """
#     # Ensure numpy arrays
#     y_true = np.asarray(y_true).squeeze()
#     y_score = np.asarray(y_score).squeeze()
    
#     # Sort by score in descending order
#     sort_indices = np.argsort(y_score)[::-1]
#     y_true_sorted = y_true[sort_indices]
    
#     total_positives = np.sum(y_true)
#     total_samples = len(y_true_sorted)
#     metrics = {}
    
#     for k_pct in k_percentages:
#         # Calculate absolute K from percentage
#         k = int(np.ceil(total_samples * (k_pct / 100.0)))
#         print(k," ")

#         if k > total_samples:
#             k = total_samples 
        
#         if k == 0:
#             continue
            
#         top_k_labels = y_true_sorted[:k]
#         num_positives_in_top_k = np.sum(top_k_labels)
        
#         # Use a metric name key
#         metric_key = f"at_{k_pct}_pct"

#         # Precision@K
#         metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
        
#         # Recall@K
#         if total_positives > 0:
#             metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
#         else:
#             metrics[f'recall_{metric_key}'] = 0.0 # Handle case with no positives
        
#         # NDCG@K
#         # ndcg_score expects (n_samples, n_labels) -> (1, n_graphs)
#         true_relevance = y_true.reshape(1, -1)
#         pred_scores = y_score.reshape(1, -1)
#         metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

#     return metrics

# # =====================================================================================
# #  ADDED: Plotting Function
# # =====================================================================================

# def save_ranking_plots(summary_data, k_percentages, output_dir):
#     """
#     Generates and saves plots for Precision, Recall, and NDCG at K%.
#     Plots show the mean +- standard deviation.
#     """
#     print(f"Generating plots in {output_dir}...")
    
#     metric_bases = ['precision', 'recall', 'ndcg']
#     for metric_base in metric_bases:
#         try:
#             # Extract means and stds from the summary dictionary
#             means = [summary_data.get(f'{metric_base}_at_{k}_pct_mean', np.nan) for k in k_percentages]
#             stds = [summary_data.get(f'{metric_base}_at_{k}_pct_std', np.nan) for k in k_percentages]
            
#             # Convert to numpy arrays for plotting
#             means = np.array(means)
#             stds = np.array(stds)
            
#             # Remove NaN values (in case some K% failed or wasn't calculated)
#             valid_indices = ~np.isnan(means)
#             if not np.any(valid_indices):
#                 print(f"Skipping plot for {metric_base}: No valid data found.")
#                 continue

#             # Filter data
#             x_axis = np.array(k_percentages)[valid_indices]
#             plot_means = means[valid_indices]
#             plot_stds = stds[valid_indices]

#             plt.figure(figsize=(10, 6))
#             # Plot the mean line
#             plt.plot(x_axis, plot_means, 'o-', label='Mean')
#             # Plot the standard deviation band
#             plt.fill_between(x_axis, plot_means - plot_stds, plot_means + plot_stds,
#                              color='blue', alpha=0.2, label='Mean ± 1 Std Dev')
            
#             plt.title(f'{metric_base.capitalize()} vs. Top K%')
#             plt.xlabel('Top K (%)')
#             plt.ylabel(f'Mean {metric_base.capitalize()}')
#             plt.grid(True, linestyle='--', alpha=0.6)
#             plt.legend()
#             plt.xticks(x_axis[::2]) # Show every other K% tick to avoid crowding
            
#             # Save the plot
#             plot_filename = os.path.join(output_dir, f'{metric_base}_vs_k_percent.png')
#             plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#             plt.close()
#             print(f"Saved {plot_filename}")

#         except Exception as e:
#             print(f"Warning: Could not generate plot for {metric_base}. Error: {e}")


# # =====================================================================================
# # (Loss functions and train function removed)
# # =====================================================================================

# def fix_seed(seed):
#     """Sets the seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # =====================================================================================
# #  ADDED: Parameter String Parser
# # =====================================================================================

# def parse_param_string_to_args(param_string, base_args):
#     """
#     Parses an experiment folder name (param_string) and updates
#     a copy of the base_args namespace.
#     """
#     args = argparse.Namespace(**vars(base_args)) # Create a copy
#     parts = param_string.split('_')
    
#     # Define mappings from short key in folder name to arg name
#     key_map = {
#         'lr': 'lr', 'pool': 'pool', 'metric': 'metric', 'hid': 'hidden_channels',
#         'do': 'dropout', 'layers': 'num_layers', 'heads': 'nhead', 'bn': 'use_bn',
#         'res': 'use_residual', 'focal_g': 'focal_gamma', 'focal_a': 'focal_alpha'
#     }
    
#     # Define type conversions
#     type_map = {
#         'lr': float, 'pool': str, 'metric': str, 'hid': int, 'do': float,
#         'layers': int, 'heads': int, 'bn': lambda x: x.lower() == 'true',
#         'res': lambda x: x.lower() == 'true', 'focal_g': float, 'focal_a': float
#     }
    
#     # Handle boolean flags that just exist as a word
#     if 'class_weight' in parts:
#         args.use_class_weight = True
        
#     # Heuristic: if focal_g is present, focal_loss was used
#     if any(p.startswith('focal_g-') for p in parts):
#         args.use_focal_loss = True

#     # Handle key-value pairs
#     for part in parts:
#         if '-' not in part:
#             continue
        
#         try:
#             key_short, val = part.split('-', 1)
            
#             if key_short in key_map:
#                 arg_name = key_map[key_short]
#                 arg_type = type_map[key_short]
#                 setattr(args, arg_name, arg_type(val))
#         except Exception as e:
#             print(f"Warning: Could not parse arg part '{part}' from folder name. Using default. Error: {e}")
            
#     return args

# # =====================================================================================

# def main():
#     # --- MODIFIED: Argument parser only *requires* dataset, model, runs ---
#     parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification - AUTO EVALUATION')
    
#     # --- Required Args ---
#     parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ogbg-molhiv)')
#     parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gcn, gat)')
#     parser.add_argument('--runs', type=int, required=True, help='Number of runs to evaluate (e.g., 3)')

#     # --- Other args are "defaults" to be overridden by folder name ---
#     parser.add_argument('--model_family', type=str, default='gnn', help='gnn or gt')
#     parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
#     parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
#     parser.add_argument('--metric', type=str, default='rocauc', help='Metric for best model selection (default: rocauc)')
    
#     parser.add_argument('--num_layers', type=int, default=5, help='Number of GNN layers')
#     parser.add_argument('--hidden_channels', type=int, default=300, help='Number of hidden units')
#     parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
#     parser.add_argument('--pool', type=str, default='mean', help='Pooling layer (mean, sum, add, max)')
#     parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
#     parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
#     parser.add_argument('--nhead', type=int, default=1, help='Number of heads for GAT/GT')
#     parser.add_argument('--use_bn', action='store_true', default=False, help='Use batch norm')
#     parser.add_argument('--use_residual', action='store_true', default=False, help='Use residual connections')
    
#     parser.add_argument('--use_class_weight', action='store_true', default=False,
#                         help='Use inverse class frequency to weight the loss function.')
#     parser.add_argument('--use_focal_loss', action='store_true', default=False,
#                         help='Use Focal Loss instead of standard NLL/BCE Loss.')
#     parser.add_argument('--focal_gamma', type=float, default=2.0,
#                         help='Gamma focusing parameter for Focal Loss.')
#     parser.add_argument('--focal_alpha', type=float, default=0.25,
#                         help='Alpha balancing parameter for Binary Focal Loss (for OGB).')

#     # --- ADDED: Argument for filtering experiment folders ---
#     parser.add_argument('--filter', type=str, default=None, 
#                         help='Only evaluate experiment folders containing this string (e.g., "lr-0.0001").')
#     # --- END ADDED ---

#     base_args = parser.parse_args()
    
#     # --- Infer dataset_type ---
#     base_args.dataset_type = 'ogb' if 'ogb' in base_args.dataset else 'tu'
    
#     device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
#     fix_seed(base_args.seed)
    
#     # --- Data Loading (Do this ONCE) ---
#     print(f"--- Loading dataset {base_args.dataset} ---")
#     if base_args.dataset_type == 'ogb':
#         with torch.serialization.safe_globals([DataEdgeAttr]):
#             dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
#         split_idx = dataset.get_idx_split()
#         # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=base_args.batch_size, shuffle=True)
#         valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=base_args.batch_size, shuffle=False)
#         test_loader = DataLoader(dataset[split_idx["test"]], batch_size=base_args.batch_size, shuffle=False)
#         evaluator = Evaluator(name=base_args.dataset)
#         num_tasks = dataset.num_tasks
#         num_features = dataset.num_features
#         is_binary = True 
#     else: # TU Datasets
#         dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
#         if dataset.num_node_features == 0:
#             max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset if g.edge_index is not None and g.edge_index.numel() > 0)
#             dataset.transform = OneHotDegree(max_degree=max_deg)
        
#         indices = list(range(len(dataset)))
#         random.shuffle(indices) # Note: This shuffles, but we are only using test/valid loaders
#         test_size = len(dataset) // 10
#         valid_size = len(dataset) // 10
        
#         test_dataset = dataset[indices[:test_size]]
#         valid_dataset = dataset[indices[test_size:test_size + valid_size]]

#         valid_loader = DataLoader(valid_dataset, batch_size=base_args.batch_size, shuffle=False)
#         test_loader = DataLoader(test_dataset, batch_size=base_args.batch_size, shuffle=False)
#         evaluator = None
#         num_tasks = dataset.num_classes
#         num_features = dataset.num_features
#         print('num_features', num_features)
#         is_binary = dataset.num_classes == 2
#     # --- End Data Loading ---


#     # --- K Percentages (Define ONCE) ---
#     k_percentages = [1] + list(range(5, 101, 5))
#     print(f"Will calculate ranking metrics for K={k_percentages} (as percentages)")


#     # --- MODIFIED: Main evaluation loop ---
#     base_path = f'results_weighted/{base_args.dataset}/{base_args.model_name}'
#     print(f"\nScanning for experiment directories in: {base_path}")
    
#     try:
#         # Find all subdirectories
#         experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
#     except FileNotFoundError:
#         print(f"ERROR: Base directory not found: {base_path}")
#         print("Please check your --dataset and --model_name arguments.")
#         return

#     if not experiment_dirs:
#         print(f"No experiment directories found in {base_path}.")
#         return

#     # Filter out summary/topk folders to avoid loops
#     experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
    
#     # --- ADDED: Optional filter for experiment names ---
#     if base_args.filter:
#         print(f"--- Applying filter: '{base_args.filter}' ---")
#         experiment_dirs = [d for d in experiment_dirs if base_args.filter in d]
#         print(f"Found {len(experiment_dirs)} matching experiment(s).")
#     else:
#         print(f"Found {len(experiment_dirs)} experiment(s) to evaluate.")
#     # --- END ADDED ---


#     # --- Outer loop for each experiment folder ---
#     for param_string in experiment_dirs:
#         experiment_path = os.path.join(base_path, param_string)
#         print(f"\n{'='*50}\nProcessing Experiment: {param_string}\n{'='*50}")
        
#         # --- Parse the folder name to get the *actual* args for this run ---
#         try:
#             args = parse_param_string_to_args(param_string, base_args)
#             print(f"Parsed args: hidden={args.hidden_channels}, layers={args.num_layers}, pool={args.pool}, lr={args.lr}")
#         except Exception as e:
#             print(f"ERROR: Failed to parse param string '{param_string}'. Skipping. Error: {e}")
#             continue

#         # --- Metric Tracking Setup (Reset for each experiment) ---
#         all_best_valid = [] 
#         all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
#         all_ranking_metrics = {f'precision_at_{k}_pct': [] for k in k_percentages}
#         all_ranking_metrics.update({f'recall_at_{k}_pct': [] for k in k_percentages})
#         all_ranking_metrics.update({f'ndcg_at_{k}_pct': [] for k in k_percentages})
#         all_reports_default, all_reports_optimal = [], []

#         # --- Inner loop for each run (1, 2, 3...) ---
#         for run in range(1, args.runs + 1):
#             print(f"\n--- Evaluating Run {run}/{args.runs} ---")
            
#             # --- Model construction (uses parsed args) ---
#             models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE}
#             ModelClass = models[args.model_name]
#             use_ogb = True if args.dataset_type == 'ogb' else False
            
#             try:
#                 # --- MODIFIED: Smartly build constructor arguments ---
                
#                 # Base args for GNN family
#                 gnn_args = {
#                     'num_features': num_features,
#                     'num_classes': num_tasks, # MODIFIED: was 'num_tasks'
#                     'hidden': args.hidden_channels,
#                     'num_layers': args.num_layers,
#                     'dropout': args.dropout,
#                     'pool': args.pool,
#                     'use_ogb_features': use_ogb,
#                     'use_bn': args.use_bn,
#                     'use_residual': args.use_residual
#                 }
                
#                 # Add 'nhead' ONLY if the model is 'gat'
#                 if args.model_name == 'gat':
#                     gnn_args['nhead'] = args.nhead
                    
#                 # The GAT model in gnn_models.py might use the 'gt' (transformer)
#                 # style constructor. We check for that as a fallback.
#                 gt_args = {
#                     'num_features': num_features,
#                     'num_classes': num_tasks, # MODIFIED: was 'num_tasks'
#                     'num_layer': args.num_layers,
#                     'hidden_channels': args.hidden_channels,
#                     'nhead': args.nhead,
#                     'dropout': args.dropout,
#                     'pool': args.pool,
#                     'use_ogb_features': use_ogb
#                 }

#                 if args.model_family == 'gnn' or args.model_name in ['gcn', 'sage']:
#                     try:
#                         # Try instantiating with GNN-style args
#                         model = ModelClass(**gnn_args)
#                     except TypeError as e:
#                         # If GAT was rewritten to use 'gt' args, this will fail
#                         print(f"Warning: GNN-style constructor failed ({e}). Trying GT-style.")
#                         # This is a fallback in case GAT now follows the GT constructor signature
#                         if args.model_name == 'gat':
#                              model = ModelClass(**gt_args)
#                         else:
#                              raise e # Re-raise if it wasn't GAT
#                 else: # Graph Transformer
#                      model = ModelClass(**gt_args)
#                 # --- END MODIFIED SECTION ---

#             except Exception as e:
#                 print(f"ERROR: Failed to construct model for {param_string}. Skipping run. Error: {e}")
#                 continue

#             # --- Load Saved Model (uses experiment_path) ---
#             out_dir = os.path.join(experiment_path, f'run_{run}') # Path for this specific run
#             model_path = os.path.join(out_dir, 'model.pt')

#             if not os.path.exists(model_path):
#                 print(f"ERROR: Model file not found at {model_path}")
#                 print(f"Skipping Run {run}.")
#                 continue
                
#             print(f"Loading model from: {model_path}")
#             model.load_state_dict(torch.load(model_path, map_location=device))
#             model.to(device)
#             model.eval() 
#             # --- END MODIFIED SECTION ---

#             # --- Final Evaluation (Identical to original script) ---
#             y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
#             y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

#             report_optimal = None
#             ranking_metrics_run = {}

#             if is_binary:
#                 if y_valid_pred.shape[1] == 1:
#                     valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy()
#                 else:
#                     valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
                
#                 valid_true = y_valid_true.cpu().numpy().squeeze()
#                 best_f1 = -1
#                 best_threshold = 0.5
#                 for threshold in np.linspace(0, 1, 100):
#                     f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
#                     if f1 > best_f1:
#                         best_f1, best_threshold = f1, threshold
                
#                 if y_test_pred.shape[1] == 1:
#                     test_probs_np = torch.sigmoid(y_test_pred).squeeze().cpu().numpy()
#                 else:
#                     test_probs_np = torch.softmax(y_test_pred, dim=1)[:, 1].cpu().numpy()

#                 y_test_true_np = y_test_true.cpu().numpy().squeeze()
                
#                 # --- Calculate Ranking Metrics ---
#                 ranking_metrics_run = calculate_ranking_metrics(y_test_true_np, test_probs_np, k_percentages)
#                 print(f"Run {run} Ranking Metrics (Top 1%): P={ranking_metrics_run.get('precision_at_1_pct', 0):.4f}, R={ranking_metrics_run.get('recall_at_1_pct', 0):.4f}")

#                 y_pred_optimal_np = (test_probs_np >= best_threshold).astype(int)
                
#                 test_acc_opt = accuracy_score(y_test_true_np, y_pred_optimal_np)
#                 test_bal_acc_opt = balanced_accuracy_score(y_test_true_np, y_pred_optimal_np)
#                 report_optimal = classification_report(y_test_true_np, y_pred_optimal_np, output_dict=True, zero_division=0)
                
#                 test_metrics_default['acc'] = test_acc_opt
#                 test_metrics_default['balacc'] = test_bal_acc_opt

#             # --- Aggregating and Saving Run Results ---
#             acc = test_metrics_default['acc']
#             bal_acc = test_metrics_default['balacc']
#             roc_auc = test_metrics_default['rocauc']
#             pr_auc = test_metrics_default.get('prauc', test_metrics_default.get('ap', 0))
            
#             for key, value in ranking_metrics_run.items():
#                 metric_base = key.split('_at_')[0]
#                 k_val = key.split('_at_')[1]
#                 full_key = f"{metric_base}_at_{k_val}"

#                 if full_key in all_ranking_metrics:
#                     all_ranking_metrics[full_key].append(value)

#             y_pred_default_np = torch.argmax(y_test_pred, dim=1).cpu().numpy() if y_test_pred.shape[1] > 1 else (y_test_pred.cpu().numpy() > 0).astype(int)
#             y_test_true_np = y_test_true.cpu().numpy()
#             report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
            
#             all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
#             all_reports_default.append(report_default)
#             if report_optimal: all_reports_optimal.append(report_optimal)

#             # --- We still update the per-run metrics.json in the run_{run} folder ---
#             os.makedirs(out_dir, exist_ok=True)
            
#             run_metrics = {'accuracy': acc, 'balanced_accuracy': bal_acc, 
#                              'auc_roc': roc_auc, 'auc_pr': pr_auc,
#                              'classification_report_default': report_default, 
#                              'classification_report_optimal': report_optimal}
            
#             run_metrics.update(ranking_metrics_run) # Add the new ranking metrics

#             # Try to load best_validation_metric from existing file
#             try:
#                 with open(os.path.join(out_dir, 'metrics.json'), 'r') as f:
#                     existing_metrics = json.load(f)
#                     if 'best_validation_metric' in existing_metrics:
#                         run_metrics['best_validation_metric'] = existing_metrics['best_validation_metric']
#                         all_best_valid.append(existing_metrics['best_validation_metric'])
#             except FileNotFoundError:
#                 pass # No existing file

#             # Overwrite the metrics.json file with all new data
#             with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
#                 json.dump(run_metrics, f, indent=2)
        
#         # --- End of inner 'run' loop ---

#         # --- Final Summary (for this experiment) ---
        
#         # --- Save summary to 'topk_percent' folder *inside this experiment_path* ---
#         summary_dir = os.path.join(experiment_path, 'topk_percent')
#         os.makedirs(summary_dir, exist_ok=True)
#         print(f"\nSaving final summary and plots for {param_string} to: {summary_dir}")
        
#         # --- Create summary with ONLY ranking metrics ---
#         summary = {}

#         # Add ranking metrics to final summary
#         for key, values in all_ranking_metrics.items():
#             if values: # Only add if we have data
#                 summary[f'{key}_mean'] = np.mean(values)
#                 summary[f'{key}_std'] = np.std(values)

#         with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         # --- Call the plotting function ---
#         save_ranking_plots(summary, k_percentages, summary_dir)

#         print(f"\n{'='*40}\nFINAL SUMMARY (Top K%) for {param_string}\n{'='*40}")
#         print(json.dumps(summary, indent=2)) # MODIFIED: Removed extra 'f' argument
    
#     # --- End of outer 'experiment' loop ---
#     print(f"\n{'='*50}\nAll experiments evaluated.\n{'='*50}")


# if __name__ == "__main__":
#     main()



# # import argparse
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np
# # import os
# # import json
# # import random
# # import glob

# # # --- ADDED: Matplotlib for plotting ---
# # import matplotlib
# # matplotlib.use('Agg') # Use 'Agg' backend for headless servers
# # import matplotlib.pyplot as plt

# # # --- Import all required metrics ---
# # from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score, ndcg_score

# # from torch_geometric.loader import DataLoader
# # from torch_geometric.datasets import TUDataset
# # from torch_geometric.transforms import OneHotDegree
# # from torch_geometric.utils import degree
# # from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# # # from parse_graphclass import parser_add_main_args # MODIFIED: We define args locally
# # from gnn_models import GCN, GAT, SAGE, GIN
# # # from ignore_gt_models import Graphormer, GraphiT, GPS
# # from training_utils import evaluate_graphclass
# # from torch_geometric.data.storage import GlobalStorage
# # from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# # torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# # # =====================================================================================
# # #  MODIFIED: Ranking Metrics Helper (now uses percentages)
# # # =====================================================================================

# # def calculate_ranking_metrics(y_true, y_score, k_percentages):
# #     """
# #     Calculates Precision@K, Recall@K, and NDCG@K for a list of K percentages.
# #     Assumes y_true are 0/1 relevance labels and y_score are prediction scores.
# #     """
# #     # Ensure numpy arrays
# #     y_true = np.asarray(y_true).squeeze()
# #     y_score = np.asarray(y_score).squeeze()
    
# #     # Sort by score in descending order
# #     sort_indices = np.argsort(y_score)[::-1]
# #     y_true_sorted = y_true[sort_indices]
    
# #     total_positives = np.sum(y_true)
# #     total_samples = len(y_true_sorted)
# #     metrics = {}
    
# #     for k_pct in k_percentages:
# #         # Calculate absolute K from percentage
# #         k = int(np.ceil(total_samples * (k_pct / 100.0)))

# #         if k > total_samples:
# #             k = total_samples 
        
# #         if k == 0:
# #             continue
            
# #         top_k_labels = y_true_sorted[:k]
# #         num_positives_in_top_k = np.sum(top_k_labels)
        
# #         # Use a metric name key
# #         metric_key = f"at_{k_pct}_pct"

# #         # Precision@K
# #         metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
        
# #         # Recall@K
# #         if total_positives > 0:
# #             metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
# #         else:
# #             metrics[f'recall_{metric_key}'] = 0.0 # Handle case with no positives
        
# #         # NDCG@K
# #         # ndcg_score expects (n_samples, n_labels) -> (1, n_graphs)
# #         true_relevance = y_true.reshape(1, -1)
# #         pred_scores = y_score.reshape(1, -1)
# #         metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

# #     return metrics

# # # =====================================================================================
# # #  ADDED: Plotting Function
# # # =====================================================================================

# # def save_ranking_plots(summary_data, k_percentages, output_dir):
# #     """
# #     Generates and saves plots for Precision, Recall, and NDCG at K%.
# #     Plots show the mean +- standard deviation.
# #     """
# #     print(f"Generating plots in {output_dir}...")
    
# #     metric_bases = ['precision', 'recall', 'ndcg']
# #     for metric_base in metric_bases:
# #         try:
# #             # Extract means and stds from the summary dictionary
# #             means = [summary_data.get(f'{metric_base}_at_{k}_pct_mean', np.nan) for k in k_percentages]
# #             stds = [summary_data.get(f'{metric_base}_at_{k}_pct_std', np.nan) for k in k_percentages]
            
# #             # Convert to numpy arrays for plotting
# #             means = np.array(means)
# #             stds = np.array(stds)
            
# #             # Remove NaN values (in case some K% failed or wasn't calculated)
# #             valid_indices = ~np.isnan(means)
# #             if not np.any(valid_indices):
# #                 print(f"Skipping plot for {metric_base}: No valid data found.")
# #                 continue

# #             # Filter data
# #             x_axis = np.array(k_percentages)[valid_indices]
# #             plot_means = means[valid_indices]
# #             plot_stds = stds[valid_indices]

# #             plt.figure(figsize=(10, 6))
# #             # Plot the mean line
# #             plt.plot(x_axis, plot_means, 'o-', label='Mean')
# #             # Plot the standard deviation band
# #             plt.fill_between(x_axis, plot_means - plot_stds, plot_means + plot_stds,
# #                              color='blue', alpha=0.2, label='Mean ± 1 Std Dev')
            
# #             plt.title(f'{metric_base.capitalize()} vs. Top K%')
# #             plt.xlabel('Top K (%)')
# #             plt.ylabel(f'Mean {metric_base.capitalize()}')
# #             plt.grid(True, linestyle='--', alpha=0.6)
# #             plt.legend()
# #             plt.xticks(x_axis[::2]) # Show every other K% tick to avoid crowding
            
# #             # Save the plot
# #             plot_filename = os.path.join(output_dir, f'{metric_base}_vs_k_percent.png')
# #             plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
# #             plt.close()
# #             print(f"Saved {plot_filename}")

# #         except Exception as e:
# #             print(f"Warning: Could not generate plot for {metric_base}. Error: {e}")


# # # =====================================================================================
# # # (Loss functions and train function removed)
# # # =====================================================================================

# # def fix_seed(seed):
# #     """Sets the seed for reproducibility."""
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)

# # # =====================================================================================
# # #  ADDED: Parameter String Parser
# # # =====================================================================================

# # def parse_param_string_to_args(param_string, base_args):
# #     """
# #     Parses an experiment folder name (param_string) and updates
# #     a copy of the base_args namespace.
# #     """
# #     args = argparse.Namespace(**vars(base_args)) # Create a copy
# #     parts = param_string.split('_')
    
# #     # Define mappings from short key in folder name to arg name
# #     key_map = {
# #         'lr': 'lr', 'pool': 'pool', 'metric': 'metric', 'hid': 'hidden_channels',
# #         'do': 'dropout', 'layers': 'num_layers', 'heads': 'nhead', 'bn': 'use_bn',
# #         'res': 'use_residual', 'focal_g': 'focal_gamma', 'focal_a': 'focal_alpha'
# #     }
    
# #     # Define type conversions
# #     type_map = {
# #         'lr': float, 'pool': str, 'metric': str, 'hid': int, 'do': float,
# #         'layers': int, 'heads': int, 'bn': lambda x: x.lower() == 'true',
# #         'res': lambda x: x.lower() == 'true', 'focal_g': float, 'focal_a': float
# #     }
    
# #     # Handle boolean flags that just exist as a word
# #     if 'class_weight' in parts:
# #         args.use_class_weight = True
        
# #     # Heuristic: if focal_g is present, focal_loss was used
# #     if any(p.startswith('focal_g-') for p in parts):
# #         args.use_focal_loss = True

# #     # Handle key-value pairs
# #     for part in parts:
# #         if '-' not in part:
# #             continue
        
# #         try:
# #             key_short, val = part.split('-', 1)
            
# #             if key_short in key_map:
# #                 arg_name = key_map[key_short]
# #                 arg_type = type_map[key_short]
# #                 setattr(args, arg_name, arg_type(val))
# #         except Exception as e:
# #             print(f"Warning: Could not parse arg part '{part}' from folder name. Using default. Error: {e}")
            
# #     return args

# # # =====================================================================================

# # def main():
# #     # --- MODIFIED: Argument parser only *requires* dataset, model, runs ---
# #     parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification - AUTO EVALUATION')
    
# #     # --- Required Args ---
# #     parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ogbg-molhiv)')
# #     parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gcn, gat)')
# #     parser.add_argument('--runs', type=int, required=True, help='Number of runs to evaluate (e.g., 3)')

# #     # --- Other args are "defaults" to be overridden by folder name ---
# #     parser.add_argument('--model_family', type=str, default='gnn', help='gnn or gt')
# #     parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
# #     parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
# #     parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
# #     parser.add_argument('--metric', type=str, default='rocauc', help='Metric for best model selection (default: rocauc)')
    
# #     parser.add_argument('--num_layers', type=int, default=5, help='Number of GNN layers')
# #     parser.add_argument('--hidden_channels', type=int, default=300, help='Number of hidden units')
# #     parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
# #     parser.add_argument('--pool', type=str, default='mean', help='Pooling layer (mean, sum, add, max)')
# #     parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
# #     parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
# #     parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
# #     parser.add_argument('--nhead', type=int, default=1, help='Number of heads for GAT/GT')
# #     parser.add_argument('--use_bn', action='store_true', default=False, help='Use batch norm')
# #     parser.add_argument('--use_residual', action='store_true', default=False, help='Use residual connections')
    
# #     parser.add_argument('--use_class_weight', action='store_true', default=False,
# #                         help='Use inverse class frequency to weight the loss function.')
# #     parser.add_argument('--use_focal_loss', action='store_true', default=False,
# #                         help='Use Focal Loss instead of standard NLL/BCE Loss.')
# #     parser.add_argument('--focal_gamma', type=float, default=2.0,
# #                         help='Gamma focusing parameter for Focal Loss.')
# #     parser.add_argument('--focal_alpha', type=float, default=0.25,
# #                         help='Alpha balancing parameter for Binary Focal Loss (for OGB).')

# #     base_args = parser.parse_args()
    
# #     # --- Infer dataset_type ---
# #     base_args.dataset_type = 'ogb' if 'ogb' in base_args.dataset else 'tu'
    
# #     device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
# #     fix_seed(base_args.seed)
    
# #     # --- Data Loading (Do this ONCE) ---
# #     print(f"--- Loading dataset {base_args.dataset} ---")
# #     if base_args.dataset_type == 'ogb':
# #         with torch.serialization.safe_globals([DataEdgeAttr]):
# #             dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
# #         split_idx = dataset.get_idx_split()
# #         # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=base_args.batch_size, shuffle=True)
# #         valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=base_args.batch_size, shuffle=False)
# #         test_loader = DataLoader(dataset[split_idx["test"]], batch_size=base_args.batch_size, shuffle=False)
# #         evaluator = Evaluator(name=base_args.dataset)
# #         num_tasks = dataset.num_tasks
# #         num_features = dataset.num_features
# #         is_binary = True 
# #     else: # TU Datasets
# #         dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
# #         if dataset.num_node_features == 0:
# #             max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset if g.edge_index is not None and g.edge_index.numel() > 0)
# #             dataset.transform = OneHotDegree(max_degree=max_deg)
        
# #         indices = list(range(len(dataset)))
# #         random.shuffle(indices) # Note: This shuffles, but we are only using test/valid loaders
# #         test_size = len(dataset) // 10
# #         valid_size = len(dataset) // 10
        
# #         test_dataset = dataset[indices[:test_size]]
# #         valid_dataset = dataset[indices[test_size:test_size + valid_size]]

# #         valid_loader = DataLoader(valid_dataset, batch_size=base_args.batch_size, shuffle=False)
# #         test_loader = DataLoader(test_dataset, batch_size=base_args.batch_size, shuffle=False)
# #         evaluator = None
# #         num_tasks = dataset.num_classes
# #         num_features = dataset.num_features
# #         print('num_features', num_features)
# #         is_binary = dataset.num_classes == 2
# #     # --- End Data Loading ---


# #     # --- K Percentages (Define ONCE) ---
# #     k_percentages = [1] + list(range(5, 101, 5))
# #     print(f"Will calculate ranking metrics for K={k_percentages} (as percentages)")


# #     # --- MODIFIED: Main evaluation loop ---
# #     base_path = f'results_weighted/{base_args.dataset}/{base_args.model_name}'
# #     print(f"\nScanning for experiment directories in: {base_path}")
    
# #     try:
# #         # Find all subdirectories
# #         experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
# #     except FileNotFoundError:
# #         print(f"ERROR: Base directory not found: {base_path}")
# #         print("Please check your --dataset and --model_name arguments.")
# #         return

# #     if not experiment_dirs:
# #         print(f"No experiment directories found in {base_path}.")
# #         return

# #     # Filter out summary/topk folders to avoid loops
# #     experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
# #     print(f"Found {len(experiment_dirs)} experiment(s) to evaluate.")


# #     # --- Outer loop for each experiment folder ---
# #     for param_string in experiment_dirs:
# #         experiment_path = os.path.join(base_path, param_string)
# #         print(f"\n{'='*50}\nProcessing Experiment: {param_string}\n{'='*50}")
        
# #         # --- Parse the folder name to get the *actual* args for this run ---
# #         try:
# #             args = parse_param_string_to_args(param_string, base_args)
# #             print(f"Parsed args: hidden={args.hidden_channels}, layers={args.num_layers}, pool={args.pool}, lr={args.lr}")
# #         except Exception as e:
# #             print(f"ERROR: Failed to parse param string '{param_string}'. Skipping. Error: {e}")
# #             continue

# #         # --- Metric Tracking Setup (Reset for each experiment) ---
# #         all_best_valid = [] 
# #         all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
# #         all_ranking_metrics = {f'precision_at_{k}_pct': [] for k in k_percentages}
# #         all_ranking_metrics.update({f'recall_at_{k}_pct': [] for k in k_percentages})
# #         all_ranking_metrics.update({f'ndcg_at_{k}_pct': [] for k in k_percentages})
# #         all_reports_default, all_reports_optimal = [], []

# #         # --- Inner loop for each run (1, 2, 3...) ---
# #         for run in range(1, args.runs + 1):
# #             print(f"\n--- Evaluating Run {run}/{args.runs} ---")
            
# #             # --- Model construction (uses parsed args) ---
# #             models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE}
# #             ModelClass = models[args.model_name]
# #             use_ogb = True if args.dataset_type == 'ogb' else False
            
# #             try:
# #                 # --- MODIFIED: Smartly build constructor arguments ---
                
# #                 # Base args for GNN family
# #                 gnn_args = {
# #                     'num_features': num_features,
# #                     'num_tasks': num_tasks,
# #                     'hidden': args.hidden_channels,
# #                     'num_layers': args.num_layers,
# #                     'dropout': args.dropout,
# #                     'pool': args.pool,
# #                     'use_ogb_features': use_ogb,
# #                     'use_bn': args.use_bn,
# #                     'use_residual': args.use_residual
# #                 }
                
# #                 # Add 'nhead' ONLY if the model is 'gat'
# #                 if args.model_name == 'gat':
# #                     gnn_args['nhead'] = args.nhead
                    
# #                 # The GAT model in gnn_models.py might use the 'gt' (transformer)
# #                 # style constructor. We check for that as a fallback.
# #                 gt_args = {
# #                     'num_features': num_features,
# #                     'num_tasks': num_tasks,
# #                     'num_layer': args.num_layers,
# #                     'hidden_channels': args.hidden_channels,
# #                     'nhead': args.nhead,
# #                     'dropout': args.dropout,
# #                     'pool': args.pool,
# #                     'use_ogb_features': use_ogb
# #                 }

# #                 if args.model_family == 'gnn' or args.model_name in ['gcn', 'sage']:
# #                     try:
# #                         # Try instantiating with GNN-style args
# #                         model = ModelClass(**gnn_args)
# #                     except TypeError as e:
# #                         # If GAT was rewritten to use 'gt' args, this will fail
# #                         print(f"Warning: GNN-style constructor failed ({e}). Trying GT-style.")
# #                         # This is a fallback in case GAT now follows the GT constructor signature
# #                         if args.model_name == 'gat':
# #                              model = ModelClass(**gt_args)
# #                         else:
# #                              raise e # Re-raise if it wasn't GAT
# #                 else: # Graph Transformer
# #                      model = ModelClass(**gt_args)
# #                 # --- END MODIFIED SECTION ---

# #             except Exception as e:
# #                 print(f"ERROR: Failed to construct model for {param_string}. Skipping run. Error: {e}")
# #                 continue

# #             # --- Load Saved Model (uses experiment_path) ---
# #             out_dir = os.path.join(experiment_path, f'run_{run}') # Path for this specific run
# #             model_path = os.path.join(out_dir, 'model.pt')

# #             if not os.path.exists(model_path):
# #                 print(f"ERROR: Model file not found at {model_path}")
# #                 print(f"Skipping Run {run}.")
# #                 continue
                
# #             print(f"Loading model from: {model_path}")
# #             model.load_state_dict(torch.load(model_path, map_location=device))
# #             model.to(device)
# #             model.eval() 
# #             # --- END MODIFIED SECTION ---

# #             # --- Final Evaluation (Identical to original script) ---
# #             y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
# #             y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

# #             report_optimal = None
# #             ranking_metrics_run = {}

# #             if is_binary:
# #                 if y_valid_pred.shape[1] == 1:
# #                     valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy()
# #                 else:
# #                     valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
                
# #                 valid_true = y_valid_true.cpu().numpy().squeeze()
# #                 best_f1 = -1
# #                 best_threshold = 0.5
# #                 for threshold in np.linspace(0, 1, 100):
# #                     f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
# #                     if f1 > best_f1:
# #                         best_f1, best_threshold = f1, threshold
                
# #                 if y_test_pred.shape[1] == 1:
# #                     test_probs_np = torch.sigmoid(y_test_pred).squeeze().cpu().numpy()
# #                 else:
# #                     test_probs_np = torch.softmax(y_test_pred, dim=1)[:, 1].cpu().numpy()

# #                 y_test_true_np = y_test_true.cpu().numpy().squeeze()
                
# #                 # --- Calculate Ranking Metrics ---
# #                 ranking_metrics_run = calculate_ranking_metrics(y_test_true_np, test_probs_np, k_percentages)
# #                 print(f"Run {run} Ranking Metrics (Top 1%): P={ranking_metrics_run.get('precision_at_1_pct', 0):.4f}, R={ranking_metrics_run.get('recall_at_1_pct', 0):.4f}")

# #                 y_pred_optimal_np = (test_probs_np >= best_threshold).astype(int)
                
# #                 test_acc_opt = accuracy_score(y_test_true_np, y_pred_optimal_np)
# #                 test_bal_acc_opt = balanced_accuracy_score(y_test_true_np, y_pred_optimal_np)
# #                 report_optimal = classification_report(y_test_true_np, y_pred_optimal_np, output_dict=True, zero_division=0)
                
# #                 test_metrics_default['acc'] = test_acc_opt
# #                 test_metrics_default['balacc'] = test_bal_acc_opt

# #             # --- Aggregating and Saving Run Results ---
# #             acc = test_metrics_default['acc']
# #             bal_acc = test_metrics_default['balacc']
# #             roc_auc = test_metrics_default['rocauc']
# #             pr_auc = test_metrics_default.get('prauc', test_metrics_default.get('ap', 0))
            
# #             for key, value in ranking_metrics_run.items():
# #                 metric_base = key.split('_at_')[0]
# #                 k_val = key.split('_at_')[1]
# #                 full_key = f"{metric_base}_at_{k_val}"

# #                 if full_key in all_ranking_metrics:
# #                     all_ranking_metrics[full_key].append(value)

# #             y_pred_default_np = torch.argmax(y_test_pred, dim=1).cpu().numpy() if y_test_pred.shape[1] > 1 else (y_test_pred.cpu().numpy() > 0).astype(int)
# #             y_test_true_np = y_test_true.cpu().numpy()
# #             report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
            
# #             all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
# #             all_reports_default.append(report_default)
# #             if report_optimal: all_reports_optimal.append(report_optimal)

# #             # --- We still update the per-run metrics.json in the run_{run} folder ---
# #             os.makedirs(out_dir, exist_ok=True)
            
# #             run_metrics = {'accuracy': acc, 'balanced_accuracy': bal_acc, 
# #                              'auc_roc': roc_auc, 'auc_pr': pr_auc,
# #                              'classification_report_default': report_default, 
# #                              'classification_report_optimal': report_optimal}
            
# #             run_metrics.update(ranking_metrics_run) # Add the new ranking metrics

# #             # Try to load best_validation_metric from existing file
# #             try:
# #                 with open(os.path.join(out_dir, 'metrics.json'), 'r') as f:
# #                     existing_metrics = json.load(f)
# #                     if 'best_validation_metric' in existing_metrics:
# #                         run_metrics['best_validation_metric'] = existing_metrics['best_validation_metric']
# #                         all_best_valid.append(existing_metrics['best_validation_metric'])
# #             except FileNotFoundError:
# #                 pass # No existing file

# #             # Overwrite the metrics.json file with all new data
# #             with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
# #                 json.dump(run_metrics, f, indent=2)
        
# #         # --- End of inner 'run' loop ---

# #         # --- Final Summary (for this experiment) ---
        
# #         # --- Save summary to 'topk_percent' folder *inside this experiment_path* ---
# #         summary_dir = os.path.join(experiment_path, 'topk_percent')
# #         os.makedirs(summary_dir, exist_ok=True)
# #         print(f"\nSaving final summary and plots for {param_string} to: {summary_dir}")
        
# #         # --- Create summary with ONLY ranking metrics ---
# #         summary = {}

# #         # Add ranking metrics to final summary
# #         for key, values in all_ranking_metrics.items():
# #             if values: # Only add if we have data
# #                 summary[f'{key}_mean'] = np.mean(values)
# #                 summary[f'{key}_std'] = np.std(values)

# #         with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
# #             json.dump(summary, f, indent=2)
        
# #         # --- Call the plotting function ---
# #         save_ranking_plots(summary, k_percentages, summary_dir)

# #         print(f"\n{'='*40}\nFINAL SUMMARY (Top K%) for {param_string}\n{'='*40}")
# #         print(json.dumps(summary, f, indent=2))
    
# #     # --- End of outer 'experiment' loop ---
# #     print(f"\n{'='*50}\nAll experiments evaluated.\n{'='*50}")


# # if __name__ == "__main__":
# #     main()
# # # ```

# # # ### How to Run:

# # # Now, you just run one command. For example:

# # # ```bash
# # # python eval_only.py --dataset ogbg-molhiv --model_name gcn --runs 3




# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import os
# import json
# import random
# import glob

# # --- ADDED: Matplotlib for plotting ---
# import matplotlib
# matplotlib.use('Agg') # Use 'Agg' backend for headless servers
# import matplotlib.pyplot as plt

# # --- Import all required metrics ---
# from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score, ndcg_score

# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
# from torch_geometric.transforms import OneHotDegree
# from torch_geometric.utils import degree
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# # from parse_graphclass import parser_add_main_args # MODIFIED: We define args locally
# from gnn_models import GCN, GAT, SAGE, GIN
# # from ignore_gt_models import Graphormer, GraphiT, GPS
# from training_utils import evaluate_graphclass
# from torch_geometric.data.storage import GlobalStorage
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# # =====================================================================================
# #  MODIFIED: Ranking Metrics Helper (now uses percentages)
# # =====================================================================================

# def calculate_ranking_metrics(y_true, y_score, k_percentages):
#     """
#     Calculates Precision@K, Recall@K, and NDCG@K for a list of K percentages.
#     Assumes y_true are 0/1 relevance labels and y_score are prediction scores.
#     """
#     # Ensure numpy arrays
#     y_true = np.asarray(y_true).squeeze()
#     y_score = np.asarray(y_score).squeeze()
    
#     # Sort by score in descending order
#     sort_indices = np.argsort(y_score)[::-1]
#     y_true_sorted = y_true[sort_indices]
    
#     total_positives = np.sum(y_true)
#     total_samples = len(y_true_sorted)
#     metrics = {}
    
#     for k_pct in k_percentages:
#         # Calculate absolute K from percentage
#         k = int(np.ceil(total_samples * (k_pct / 100.0)))
#         print(k," ")

#         if k > total_samples:
#             k = total_samples 
        
#         if k == 0:
#             continue
            
#         top_k_labels = y_true_sorted[:k]
#         num_positives_in_top_k = np.sum(top_k_labels)
        
#         # Use a metric name key
#         metric_key = f"at_{k_pct}_pct"

#         # Precision@K
#         metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
        
#         # Recall@K
#         if total_positives > 0:
#             metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
#         else:
#             metrics[f'recall_{metric_key}'] = 0.0 # Handle case with no positives
        
#         # NDCG@K
#         # ndcg_score expects (n_samples, n_labels) -> (1, n_graphs)
#         true_relevance = y_true.reshape(1, -1)
#         pred_scores = y_score.reshape(1, -1)
#         metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

#     return metrics

# # =====================================================================================
# #  ADDED: Plotting Function
# # =====================================================================================

# def save_ranking_plots(summary_data, k_percentages, output_dir):
#     """
#     Generates and saves plots for Precision, Recall, and NDCG at K%.
#     Plots show the mean +- standard deviation.
#     """
#     print(f"Generating plots in {output_dir}...")
    
#     metric_bases = ['precision', 'recall', 'ndcg']
#     for metric_base in metric_bases:
#         try:
#             # Extract means and stds from the summary dictionary
#             means = [summary_data.get(f'{metric_base}_at_{k}_pct_mean', np.nan) for k in k_percentages]
#             stds = [summary_data.get(f'{metric_base}_at_{k}_pct_std', np.nan) for k in k_percentages]
            
#             # Convert to numpy arrays for plotting
#             means = np.array(means)
#             stds = np.array(stds)
            
#             # Remove NaN values (in case some K% failed or wasn't calculated)
#             valid_indices = ~np.isnan(means)
#             if not np.any(valid_indices):
#                 print(f"Skipping plot for {metric_base}: No valid data found.")
#                 continue

#             # Filter data
#             x_axis = np.array(k_percentages)[valid_indices]
#             plot_means = means[valid_indices]
#             plot_stds = stds[valid_indices]

#             plt.figure(figsize=(10, 6))
#             # Plot the mean line
#             plt.plot(x_axis, plot_means, 'o-', label='Mean')
#             # Plot the standard deviation band
#             plt.fill_between(x_axis, plot_means - plot_stds, plot_means + plot_stds,
#                              color='blue', alpha=0.2, label='Mean ± 1 Std Dev')
            
#             plt.title(f'{metric_base.capitalize()} vs. Top K%')
#             plt.xlabel('Top K (%)')
#             plt.ylabel(f'Mean {metric_base.capitalize()}')
#             plt.grid(True, linestyle='--', alpha=0.6)
#             plt.legend()
#             plt.xticks(x_axis[::2]) # Show every other K% tick to avoid crowding
            
#             # Save the plot
#             plot_filename = os.path.join(output_dir, f'{metric_base}_vs_k_percent.png')
#             plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#             plt.close()
#             print(f"Saved {plot_filename}")

#         except Exception as e:
#             print(f"Warning: Could not generate plot for {metric_base}. Error: {e}")


# # =====================================================================================
# # (Loss functions and train function removed)
# # =====================================================================================

# def fix_seed(seed):
#     """Sets the seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # =====================================================================================
# #  ADDED: Parameter String Parser
# # =====================================================================================

# def parse_param_string_to_args(param_string, base_args):
#     """
#     Parses an experiment folder name (param_string) and updates
#     a copy of the base_args namespace.
#     """
#     args = argparse.Namespace(**vars(base_args)) # Create a copy
#     parts = param_string.split('_')
    
#     # Define mappings from short key in folder name to arg name
#     key_map = {
#         'lr': 'lr', 'pool': 'pool', 'metric': 'metric', 'hid': 'hidden_channels',
#         'do': 'dropout', 'layers': 'num_layers', 'heads': 'nhead', 'bn': 'use_bn',
#         'res': 'use_residual', 'focal_g': 'focal_gamma', 'focal_a': 'focal_alpha'
#     }
    
#     # Define type conversions
#     type_map = {
#         'lr': float, 'pool': str, 'metric': str, 'hid': int, 'do': float,
#         'layers': int, 'heads': int, 'bn': lambda x: x.lower() == 'true',
#         'res': lambda x: x.lower() == 'true', 'focal_g': float, 'focal_a': float
#     }
    
#     # Handle boolean flags that just exist as a word
#     if 'class_weight' in parts:
#         args.use_class_weight = True
        
#     # Heuristic: if focal_g is present, focal_loss was used
#     if any(p.startswith('focal_g-') for p in parts):
#         args.use_focal_loss = True

#     # Handle key-value pairs
#     for part in parts:
#         if '-' not in part:
#             continue
        
#         try:
#             key_short, val = part.split('-', 1)
            
#             if key_short in key_map:
#                 arg_name = key_map[key_short]
#                 arg_type = type_map[key_short]
#                 setattr(args, arg_name, arg_type(val))
#         except Exception as e:
#             print(f"Warning: Could not parse arg part '{part}' from folder name. Using default. Error: {e}")
            
#     return args

# # =====================================================================================

# def main():
#     # --- MODIFIED: Argument parser only *requires* dataset, model, runs ---
#     parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification - AUTO EVALUATION')
    
#     # --- Required Args ---
#     parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ogbg-molhiv)')
#     parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gcn, gat)')
#     parser.add_argument('--runs', type=int, required=True, help='Number of runs to evaluate (e.g., 3)')

#     # --- Other args are "defaults" to be overridden by folder name ---
#     parser.add_argument('--model_family', type=str, default='gnn', help='gnn or gt')
#     parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
#     parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
#     parser.add_argument('--metric', type=str, default='rocauc', help='Metric for best model selection (default: rocauc)')
    
#     parser.add_argument('--num_layers', type=int, default=5, help='Number of GNN layers')
#     parser.add_argument('--hidden_channels', type=int, default=300, help='Number of hidden units')
#     parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
#     parser.add_argument('--pool', type=str, default='mean', help='Pooling layer (mean, sum, add, max)')
#     parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
#     parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
#     parser.add_argument('--nhead', type=int, default=1, help='Number of heads for GAT/GT')
#     parser.add_argument('--use_bn', action='store_true', default=False, help='Use batch norm')
#     parser.add_argument('--use_residual', action='store_true', default=False, help='Use residual connections')
    
#     parser.add_argument('--use_class_weight', action='store_true', default=False,
#                         help='Use inverse class frequency to weight the loss function.')
#     parser.add_argument('--use_focal_loss', action='store_true', default=False,
#                         help='Use Focal Loss instead of standard NLL/BCE Loss.')
#     parser.add_argument('--focal_gamma', type=float, default=2.0,
#                         help='Gamma focusing parameter for Focal Loss.')
#     parser.add_argument('--focal_alpha', type=float, default=0.25,
#                         help='Alpha balancing parameter for Binary Focal Loss (for OGB).')

#     # --- ADDED: Argument for filtering experiment folders ---
#     parser.add_argument('--filter', type=str, default=None, 
#                         help='Only evaluate experiment folders containing this string (e.g., "lr-0.0001").')
#     # --- END ADDED ---

#     base_args = parser.parse_args()
    
#     # --- Infer dataset_type ---
#     base_args.dataset_type = 'ogb' if 'ogb' in base_args.dataset else 'tu'
    
#     device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
#     fix_seed(base_args.seed)
    
#     # --- Data Loading (Do this ONCE) ---
#     print(f"--- Loading dataset {base_args.dataset} ---")
#     if base_args.dataset_type == 'ogb':
#         with torch.serialization.safe_globals([DataEdgeAttr]):
#             dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
#         split_idx = dataset.get_idx_split()
#         # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=base_args.batch_size, shuffle=True)
#         valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=base_args.batch_size, shuffle=False)
#         test_loader = DataLoader(dataset[split_idx["test"]], batch_size=base_args.batch_size, shuffle=False)
#         evaluator = Evaluator(name=base_args.dataset)
#         num_tasks = dataset.num_tasks
#         num_features = dataset.num_features
#         is_binary = True 
#     else: # TU Datasets
#         dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
#         if dataset.num_node_features == 0:
#             max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset if g.edge_index is not None and g.edge_index.numel() > 0)
#             dataset.transform = OneHotDegree(max_degree=max_deg)
        
#         indices = list(range(len(dataset)))
#         random.shuffle(indices) # Note: This shuffles, but we are only using test/valid loaders
#         test_size = len(dataset) // 10
#         valid_size = len(dataset) // 10
        
#         test_dataset = dataset[indices[:test_size]]
#         valid_dataset = dataset[indices[test_size:test_size + valid_size]]

#         valid_loader = DataLoader(valid_dataset, batch_size=base_args.batch_size, shuffle=False)
#         test_loader = DataLoader(test_dataset, batch_size=base_args.batch_size, shuffle=False)
#         evaluator = None
#         num_tasks = dataset.num_classes
#         num_features = dataset.num_features
#         print('num_features', num_features)
#         is_binary = dataset.num_classes == 2
#     # --- End Data Loading ---


#     # --- K Percentages (Define ONCE) ---
#     k_percentages = [1] + list(range(5, 101, 5))
#     print(f"Will calculate ranking metrics for K={k_percentages} (as percentages)")


#     # --- MODIFIED: Main evaluation loop ---
#     base_path = f'results_weighted/{base_args.dataset}/{base_args.model_name}'
#     print(f"\nScanning for experiment directories in: {base_path}")
    
#     try:
#         # Find all subdirectories
#         experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
#     except FileNotFoundError:
#         print(f"ERROR: Base directory not found: {base_path}")
#         print("Please check your --dataset and --model_name arguments.")
#         return

#     if not experiment_dirs:
#         print(f"No experiment directories found in {base_path}.")
#         return

#     # Filter out summary/topk folders to avoid loops
#     experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
    
#     # --- ADDED: Optional filter for experiment names ---
#     if base_args.filter:
#         print(f"--- Applying filter: '{base_args.filter}' ---")
#         experiment_dirs = [d for d in experiment_dirs if base_args.filter in d]
#         print(f"Found {len(experiment_dirs)} matching experiment(s).")
#     else:
#         print(f"Found {len(experiment_dirs)} experiment(s) to evaluate.")
#     # --- END ADDED ---


#     # --- Outer loop for each experiment folder ---
#     for param_string in experiment_dirs:
#         experiment_path = os.path.join(base_path, param_string)
#         print(f"\n{'='*50}\nProcessing Experiment: {param_string}\n{'='*50}")
        
#         # --- Parse the folder name to get the *actual* args for this run ---
#         try:
#             args = parse_param_string_to_args(param_string, base_args)
#             print(f"Parsed args: hidden={args.hidden_channels}, layers={args.num_layers}, pool={args.pool}, lr={args.lr}")
#         except Exception as e:
#             print(f"ERROR: Failed to parse param string '{param_string}'. Skipping. Error: {e}")
#             continue

#         # --- Metric Tracking Setup (Reset for each experiment) ---
#         all_best_valid = [] 
#         all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
#         all_ranking_metrics = {f'precision_at_{k}_pct': [] for k in k_percentages}
#         all_ranking_metrics.update({f'recall_at_{k}_pct': [] for k in k_percentages})
#         all_ranking_metrics.update({f'ndcg_at_{k}_pct': [] for k in k_percentages})
#         all_reports_default, all_reports_optimal = [], []

#         # --- Inner loop for each run (1, 2, 3...) ---
#         for run in range(1, args.runs + 1):
#             print(f"\n--- Evaluating Run {run}/{args.runs} ---")
            
#             # --- Model construction (uses parsed args) ---
#             models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE}
#             ModelClass = models[args.model_name]
#             use_ogb = True if args.dataset_type == 'ogb' else False
            
#             try:
#                 # --- MODIFIED: Smartly build constructor arguments ---
                
#                 # Base args for GNN family
#                 gnn_args = {
#                     'num_features': num_features,
#                     'num_classes': num_tasks, # MODIFIED: was 'num_tasks'
#                     'hidden': args.hidden_channels,
#                     'num_layers': args.num_layers,
#                     'dropout': args.dropout,
#                     'pool': args.pool,
#                     'use_ogb_features': use_ogb,
#                     'use_bn': args.use_bn,
#                     'use_residual': args.use_residual
#                 }
                
#                 # Add 'nhead' ONLY if the model is 'gat'
#                 if args.model_name == 'gat':
#                     gnn_args['nhead'] = args.nhead
                    
#                 # The GAT model in gnn_models.py might use the 'gt' (transformer)
#                 # style constructor. We check for that as a fallback.
#                 gt_args = {
#                     'num_features': num_features,
#                     'num_classes': num_tasks, # MODIFIED: was 'num_tasks'
#                     'num_layer': args.num_layers,
#                     'hidden_channels': args.hidden_channels,
#                     'nhead': args.nhead,
#                     'dropout': args.dropout,
#                     'pool': args.pool,
#                     'use_ogb_features': use_ogb
#                 }

#                 if args.model_family == 'gnn' or args.model_name in ['gcn', 'sage']:
#                     try:
#                         # Try instantiating with GNN-style args
#                         model = ModelClass(**gnn_args)
#                     except TypeError as e:
#                         # If GAT was rewritten to use 'gt' args, this will fail
#                         print(f"Warning: GNN-style constructor failed ({e}). Trying GT-style.")
#                         # This is a fallback in case GAT now follows the GT constructor signature
#                         if args.model_name == 'gat':
#                              model = ModelClass(**gt_args)
#                         else:
#                              raise e # Re-raise if it wasn't GAT
#                 else: # Graph Transformer
#                      model = ModelClass(**gt_args)
#                 # --- END MODIFIED SECTION ---

#             except Exception as e:
#                 print(f"ERROR: Failed to construct model for {param_string}. Skipping run. Error: {e}")
#                 continue

#             # --- Load Saved Model (uses experiment_path) ---
#             out_dir = os.path.join(experiment_path, f'run_{run}') # Path for this specific run
#             model_path = os.path.join(out_dir, 'model.pt')

#             if not os.path.exists(model_path):
#                 print(f"ERROR: Model file not found at {model_path}")
#                 print(f"Skipping Run {run}.")
#                 continue
                
#             print(f"Loading model from: {model_path}")
#             model.load_state_dict(torch.load(model_path, map_location=device))
#             model.to(device)
#             model.eval() 
#             # --- END MODIFIED SECTION ---

#             # --- Final Evaluation (Identical to original script) ---
#             y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
#             y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

#             report_optimal = None
#             ranking_metrics_run = {}

#             if is_binary:
#                 if y_valid_pred.shape[1] == 1:
#                     valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy()
#                 else:
#                     valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
                
#                 valid_true = y_valid_true.cpu().numpy().squeeze()
#                 best_f1 = -1
#                 best_threshold = 0.5
#                 for threshold in np.linspace(0, 1, 100):
#                     f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
#                     if f1 > best_f1:
#                         best_f1, best_threshold = f1, threshold
                
#                 if y_test_pred.shape[1] == 1:
#                     test_probs_np = torch.sigmoid(y_test_pred).squeeze().cpu().numpy()
#                 else:
#                     test_probs_np = torch.softmax(y_test_pred, dim=1)[:, 1].cpu().numpy()

#                 y_test_true_np = y_test_true.cpu().numpy().squeeze()
                
#                 # --- Calculate Ranking Metrics ---
#                 ranking_metrics_run = calculate_ranking_metrics(y_test_true_np, test_probs_np, k_percentages)
#                 print(f"Run {run} Ranking Metrics (Top 1%): P={ranking_metrics_run.get('precision_at_1_pct', 0):.4f}, R={ranking_metrics_run.get('recall_at_1_pct', 0):.4f}")

#                 y_pred_optimal_np = (test_probs_np >= best_threshold).astype(int)
                
#                 test_acc_opt = accuracy_score(y_test_true_np, y_pred_optimal_np)
#                 test_bal_acc_opt = balanced_accuracy_score(y_test_true_np, y_pred_optimal_np)
#                 report_optimal = classification_report(y_test_true_np, y_pred_optimal_np, output_dict=True, zero_division=0)
                
#                 test_metrics_default['acc'] = test_acc_opt
#                 test_metrics_default['balacc'] = test_bal_acc_opt

#             # --- Aggregating and Saving Run Results ---
#             acc = test_metrics_default['acc']
#             bal_acc = test_metrics_default['balacc']
#             roc_auc = test_metrics_default['rocauc']
#             pr_auc = test_metrics_default.get('prauc', test_metrics_default.get('ap', 0))
            
#             for key, value in ranking_metrics_run.items():
#                 metric_base = key.split('_at_')[0]
#                 k_val = key.split('_at_')[1]
#                 full_key = f"{metric_base}_at_{k_val}"

#                 if full_key in all_ranking_metrics:
#                     all_ranking_metrics[full_key].append(value)

#             y_pred_default_np = torch.argmax(y_test_pred, dim=1).cpu().numpy() if y_test_pred.shape[1] > 1 else (y_test_pred.cpu().numpy() > 0).astype(int)
#             y_test_true_np = y_test_true.cpu().numpy()
#             report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
            
#             all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
#             all_reports_default.append(report_default)
#             if report_optimal: all_reports_optimal.append(report_optimal)

#             # --- We still update the per-run metrics.json in the run_{run} folder ---
#             os.makedirs(out_dir, exist_ok=True)
            
#             run_metrics = {'accuracy': acc, 'balanced_accuracy': bal_acc, 
#                              'auc_roc': roc_auc, 'auc_pr': pr_auc,
#                              'classification_report_default': report_default, 
#                              'classification_report_optimal': report_optimal}
            
#             run_metrics.update(ranking_metrics_run) # Add the new ranking metrics

#             # Try to load best_validation_metric from existing file
#             try:
#                 with open(os.path.join(out_dir, 'metrics.json'), 'r') as f:
#                     existing_metrics = json.load(f)
#                     if 'best_validation_metric' in existing_metrics:
#                         run_metrics['best_validation_metric'] = existing_metrics['best_validation_metric']
#                         all_best_valid.append(existing_metrics['best_validation_metric'])
#             except FileNotFoundError:
#                 pass # No existing file

#             # Overwrite the metrics.json file with all new data
#             with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
#                 json.dump(run_metrics, f, indent=2)
        
#         # --- End of inner 'run' loop ---

#         # --- Final Summary (for this experiment) ---
        
#         # --- Save summary to 'topk_percent' folder *inside this experiment_path* ---
#         summary_dir = os.path.join(experiment_path, 'topk_percent')
#         os.makedirs(summary_dir, exist_ok=True)
#         print(f"\nSaving final summary and plots for {param_string} to: {summary_dir}")
        
#         # --- Create summary with ONLY ranking metrics ---
#         summary = {}

#         # Add ranking metrics to final summary
#         for key, values in all_ranking_metrics.items():
#             if values: # Only add if we have data
#                 summary[f'{key}_mean'] = np.mean(values)
#                 summary[f'{key}_std'] = np.std(values)

#         with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         # --- Call the plotting function ---
#         save_ranking_plots(summary, k_percentages, summary_dir)

#         print(f"\n{'='*40}\nFINAL SUMMARY (Top K%) for {param_string}\n{'='*40}")
#         print(json.dumps(summary, indent=2)) # MODIFIED: Removed extra 'f' argument
    
#     # --- End of outer 'experiment' loop ---
#     print(f"\n{'='*50}\nAll experiments evaluated.\n{'='*50}")


# if __name__ == "__main__":
#     main()



# # import argparse
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np
# # import os
# # import json
# # import random
# # import glob

# # # --- ADDED: Matplotlib for plotting ---
# # import matplotlib
# # matplotlib.use('Agg') # Use 'Agg' backend for headless servers
# # import matplotlib.pyplot as plt

# # # --- Import all required metrics ---
# # from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score, ndcg_score

# # from torch_geometric.loader import DataLoader
# # from torch_geometric.datasets import TUDataset
# # from torch_geometric.transforms import OneHotDegree
# # from torch_geometric.utils import degree
# # from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# # # from parse_graphclass import parser_add_main_args # MODIFIED: We define args locally
# # from gnn_models import GCN, GAT, SAGE, GIN
# # # from ignore_gt_models import Graphormer, GraphiT, GPS
# # from training_utils import evaluate_graphclass
# # from torch_geometric.data.storage import GlobalStorage
# # from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# # torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# # # =====================================================================================
# # #  MODIFIED: Ranking Metrics Helper (now uses percentages)
# # # =====================================================================================

# # def calculate_ranking_metrics(y_true, y_score, k_percentages):
# #     """
# #     Calculates Precision@K, Recall@K, and NDCG@K for a list of K percentages.
# #     Assumes y_true are 0/1 relevance labels and y_score are prediction scores.
# #     """
# #     # Ensure numpy arrays
# #     y_true = np.asarray(y_true).squeeze()
# #     y_score = np.asarray(y_score).squeeze()
    
# #     # Sort by score in descending order
# #     sort_indices = np.argsort(y_score)[::-1]
# #     y_true_sorted = y_true[sort_indices]
    
# #     total_positives = np.sum(y_true)
# #     total_samples = len(y_true_sorted)
# #     metrics = {}
    
# #     for k_pct in k_percentages:
# #         # Calculate absolute K from percentage
# #         k = int(np.ceil(total_samples * (k_pct / 100.0)))

# #         if k > total_samples:
# #             k = total_samples 
        
# #         if k == 0:
# #             continue
            
# #         top_k_labels = y_true_sorted[:k]
# #         num_positives_in_top_k = np.sum(top_k_labels)
        
# #         # Use a metric name key
# #         metric_key = f"at_{k_pct}_pct"

# #         # Precision@K
# #         metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
        
# #         # Recall@K
# #         if total_positives > 0:
# #             metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
# #         else:
# #             metrics[f'recall_{metric_key}'] = 0.0 # Handle case with no positives
        
# #         # NDCG@K
# #         # ndcg_score expects (n_samples, n_labels) -> (1, n_graphs)
# #         true_relevance = y_true.reshape(1, -1)
# #         pred_scores = y_score.reshape(1, -1)
# #         metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

# #     return metrics

# # # =====================================================================================
# # #  ADDED: Plotting Function
# # # =====================================================================================

# # def save_ranking_plots(summary_data, k_percentages, output_dir):
# #     """
# #     Generates and saves plots for Precision, Recall, and NDCG at K%.
# #     Plots show the mean +- standard deviation.
# #     """
# #     print(f"Generating plots in {output_dir}...")
    
# #     metric_bases = ['precision', 'recall', 'ndcg']
# #     for metric_base in metric_bases:
# #         try:
# #             # Extract means and stds from the summary dictionary
# #             means = [summary_data.get(f'{metric_base}_at_{k}_pct_mean', np.nan) for k in k_percentages]
# #             stds = [summary_data.get(f'{metric_base}_at_{k}_pct_std', np.nan) for k in k_percentages]
            
# #             # Convert to numpy arrays for plotting
# #             means = np.array(means)
# #             stds = np.array(stds)
            
# #             # Remove NaN values (in case some K% failed or wasn't calculated)
# #             valid_indices = ~np.isnan(means)
# #             if not np.any(valid_indices):
# #                 print(f"Skipping plot for {metric_base}: No valid data found.")
# #                 continue

# #             # Filter data
# #             x_axis = np.array(k_percentages)[valid_indices]
# #             plot_means = means[valid_indices]
# #             plot_stds = stds[valid_indices]

# #             plt.figure(figsize=(10, 6))
# #             # Plot the mean line
# #             plt.plot(x_axis, plot_means, 'o-', label='Mean')
# #             # Plot the standard deviation band
# #             plt.fill_between(x_axis, plot_means - plot_stds, plot_means + plot_stds,
# #                              color='blue', alpha=0.2, label='Mean ± 1 Std Dev')
            
# #             plt.title(f'{metric_base.capitalize()} vs. Top K%')
# #             plt.xlabel('Top K (%)')
# #             plt.ylabel(f'Mean {metric_base.capitalize()}')
# #             plt.grid(True, linestyle='--', alpha=0.6)
# #             plt.legend()
# #             plt.xticks(x_axis[::2]) # Show every other K% tick to avoid crowding
            
# #             # Save the plot
# #             plot_filename = os.path.join(output_dir, f'{metric_base}_vs_k_percent.png')
# #             plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
# #             plt.close()
# #             print(f"Saved {plot_filename}")

# #         except Exception as e:
# #             print(f"Warning: Could not generate plot for {metric_base}. Error: {e}")


# # # =====================================================================================
# # # (Loss functions and train function removed)
# # # =====================================================================================

# # def fix_seed(seed):
# #     """Sets the seed for reproducibility."""
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)

# # # =====================================================================================
# # #  ADDED: Parameter String Parser
# # # =====================================================================================

# # def parse_param_string_to_args(param_string, base_args):
# #     """
# #     Parses an experiment folder name (param_string) and updates
# #     a copy of the base_args namespace.
# #     """
# #     args = argparse.Namespace(**vars(base_args)) # Create a copy
# #     parts = param_string.split('_')
    
# #     # Define mappings from short key in folder name to arg name
# #     key_map = {
# #         'lr': 'lr', 'pool': 'pool', 'metric': 'metric', 'hid': 'hidden_channels',
# #         'do': 'dropout', 'layers': 'num_layers', 'heads': 'nhead', 'bn': 'use_bn',
# #         'res': 'use_residual', 'focal_g': 'focal_gamma', 'focal_a': 'focal_alpha'
# #     }
    
# #     # Define type conversions
# #     type_map = {
# #         'lr': float, 'pool': str, 'metric': str, 'hid': int, 'do': float,
# #         'layers': int, 'heads': int, 'bn': lambda x: x.lower() == 'true',
# #         'res': lambda x: x.lower() == 'true', 'focal_g': float, 'focal_a': float
# #     }
    
# #     # Handle boolean flags that just exist as a word
# #     if 'class_weight' in parts:
# #         args.use_class_weight = True
        
# #     # Heuristic: if focal_g is present, focal_loss was used
# #     if any(p.startswith('focal_g-') for p in parts):
# #         args.use_focal_loss = True

# #     # Handle key-value pairs
# #     for part in parts:
# #         if '-' not in part:
# #             continue
        
# #         try:
# #             key_short, val = part.split('-', 1)
            
# #             if key_short in key_map:
# #                 arg_name = key_map[key_short]
# #                 arg_type = type_map[key_short]
# #                 setattr(args, arg_name, arg_type(val))
# #         except Exception as e:
# #             print(f"Warning: Could not parse arg part '{part}' from folder name. Using default. Error: {e}")
            
# #     return args

# # # =====================================================================================

# # def main():
# #     # --- MODIFIED: Argument parser only *requires* dataset, model, runs ---
# #     parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification - AUTO EVALUATION')
    
# #     # --- Required Args ---
# #     parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ogbg-molhiv)')
# #     parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gcn, gat)')
# #     parser.add_argument('--runs', type=int, required=True, help='Number of runs to evaluate (e.g., 3)')

# #     # --- Other args are "defaults" to be overridden by folder name ---
# #     parser.add_argument('--model_family', type=str, default='gnn', help='gnn or gt')
# #     parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
# #     parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
# #     parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
# #     parser.add_argument('--metric', type=str, default='rocauc', help='Metric for best model selection (default: rocauc)')
    
# #     parser.add_argument('--num_layers', type=int, default=5, help='Number of GNN layers')
# #     parser.add_argument('--hidden_channels', type=int, default=300, help='Number of hidden units')
# #     parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
# #     parser.add_argument('--pool', type=str, default='mean', help='Pooling layer (mean, sum, add, max)')
# #     parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
# #     parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
# #     parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
# #     parser.add_argument('--nhead', type=int, default=1, help='Number of heads for GAT/GT')
# #     parser.add_argument('--use_bn', action='store_true', default=False, help='Use batch norm')
# #     parser.add_argument('--use_residual', action='store_true', default=False, help='Use residual connections')
    
# #     parser.add_argument('--use_class_weight', action='store_true', default=False,
# #                         help='Use inverse class frequency to weight the loss function.')
# #     parser.add_argument('--use_focal_loss', action='store_true', default=False,
# #                         help='Use Focal Loss instead of standard NLL/BCE Loss.')
# #     parser.add_argument('--focal_gamma', type=float, default=2.0,
# #                         help='Gamma focusing parameter for Focal Loss.')
# #     parser.add_argument('--focal_alpha', type=float, default=0.25,
# #                         help='Alpha balancing parameter for Binary Focal Loss (for OGB).')

# #     base_args = parser.parse_args()
    
# #     # --- Infer dataset_type ---
# #     base_args.dataset_type = 'ogb' if 'ogb' in base_args.dataset else 'tu'
    
# #     device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
# #     fix_seed(base_args.seed)
    
# #     # --- Data Loading (Do this ONCE) ---
# #     print(f"--- Loading dataset {base_args.dataset} ---")
# #     if base_args.dataset_type == 'ogb':
# #         with torch.serialization.safe_globals([DataEdgeAttr]):
# #             dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
# #         split_idx = dataset.get_idx_split()
# #         # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=base_args.batch_size, shuffle=True)
# #         valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=base_args.batch_size, shuffle=False)
# #         test_loader = DataLoader(dataset[split_idx["test"]], batch_size=base_args.batch_size, shuffle=False)
# #         evaluator = Evaluator(name=base_args.dataset)
# #         num_tasks = dataset.num_tasks
# #         num_features = dataset.num_features
# #         is_binary = True 
# #     else: # TU Datasets
# #         dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
# #         if dataset.num_node_features == 0:
# #             max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset if g.edge_index is not None and g.edge_index.numel() > 0)
# #             dataset.transform = OneHotDegree(max_degree=max_deg)
        
# #         indices = list(range(len(dataset)))
# #         random.shuffle(indices) # Note: This shuffles, but we are only using test/valid loaders
# #         test_size = len(dataset) // 10
# #         valid_size = len(dataset) // 10
        
# #         test_dataset = dataset[indices[:test_size]]
# #         valid_dataset = dataset[indices[test_size:test_size + valid_size]]

# #         valid_loader = DataLoader(valid_dataset, batch_size=base_args.batch_size, shuffle=False)
# #         test_loader = DataLoader(test_dataset, batch_size=base_args.batch_size, shuffle=False)
# #         evaluator = None
# #         num_tasks = dataset.num_classes
# #         num_features = dataset.num_features
# #         print('num_features', num_features)
# #         is_binary = dataset.num_classes == 2
# #     # --- End Data Loading ---


# #     # --- K Percentages (Define ONCE) ---
# #     k_percentages = [1] + list(range(5, 101, 5))
# #     print(f"Will calculate ranking metrics for K={k_percentages} (as percentages)")


# #     # --- MODIFIED: Main evaluation loop ---
# #     base_path = f'results_weighted/{base_args.dataset}/{base_args.model_name}'
# #     print(f"\nScanning for experiment directories in: {base_path}")
    
# #     try:
# #         # Find all subdirectories
# #         experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
# #     except FileNotFoundError:
# #         print(f"ERROR: Base directory not found: {base_path}")
# #         print("Please check your --dataset and --model_name arguments.")
# #         return

# #     if not experiment_dirs:
# #         print(f"No experiment directories found in {base_path}.")
# #         return

# #     # Filter out summary/topk folders to avoid loops
# #     experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]
# #     print(f"Found {len(experiment_dirs)} experiment(s) to evaluate.")


# #     # --- Outer loop for each experiment folder ---
# #     for param_string in experiment_dirs:
# #         experiment_path = os.path.join(base_path, param_string)
# #         print(f"\n{'='*50}\nProcessing Experiment: {param_string}\n{'='*50}")
        
# #         # --- Parse the folder name to get the *actual* args for this run ---
# #         try:
# #             args = parse_param_string_to_args(param_string, base_args)
# #             print(f"Parsed args: hidden={args.hidden_channels}, layers={args.num_layers}, pool={args.pool}, lr={args.lr}")
# #         except Exception as e:
# #             print(f"ERROR: Failed to parse param string '{param_string}'. Skipping. Error: {e}")
# #             continue

# #         # --- Metric Tracking Setup (Reset for each experiment) ---
# #         all_best_valid = [] 
# #         all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
# #         all_ranking_metrics = {f'precision_at_{k}_pct': [] for k in k_percentages}
# #         all_ranking_metrics.update({f'recall_at_{k}_pct': [] for k in k_percentages})
# #         all_ranking_metrics.update({f'ndcg_at_{k}_pct': [] for k in k_percentages})
# #         all_reports_default, all_reports_optimal = [], []

# #         # --- Inner loop for each run (1, 2, 3...) ---
# #         for run in range(1, args.runs + 1):
# #             print(f"\n--- Evaluating Run {run}/{args.runs} ---")
            
# #             # --- Model construction (uses parsed args) ---
# #             models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE}
# #             ModelClass = models[args.model_name]
# #             use_ogb = True if args.dataset_type == 'ogb' else False
            
# #             try:
# #                 # --- MODIFIED: Smartly build constructor arguments ---
                
# #                 # Base args for GNN family
# #                 gnn_args = {
# #                     'num_features': num_features,
# #                     'num_tasks': num_tasks,
# #                     'hidden': args.hidden_channels,
# #                     'num_layers': args.num_layers,
# #                     'dropout': args.dropout,
# #                     'pool': args.pool,
# #                     'use_ogb_features': use_ogb,
# #                     'use_bn': args.use_bn,
# #                     'use_residual': args.use_residual
# #                 }
                
# #                 # Add 'nhead' ONLY if the model is 'gat'
# #                 if args.model_name == 'gat':
# #                     gnn_args['nhead'] = args.nhead
                    
# #                 # The GAT model in gnn_models.py might use the 'gt' (transformer)
# #                 # style constructor. We check for that as a fallback.
# #                 gt_args = {
# #                     'num_features': num_features,
# #                     'num_tasks': num_tasks,
# #                     'num_layer': args.num_layers,
# #                     'hidden_channels': args.hidden_channels,
# #                     'nhead': args.nhead,
# #                     'dropout': args.dropout,
# #                     'pool': args.pool,
# #                     'use_ogb_features': use_ogb
# #                 }

# #                 if args.model_family == 'gnn' or args.model_name in ['gcn', 'sage']:
# #                     try:
# #                         # Try instantiating with GNN-style args
# #                         model = ModelClass(**gnn_args)
# #                     except TypeError as e:
# #                         # If GAT was rewritten to use 'gt' args, this will fail
# #                         print(f"Warning: GNN-style constructor failed ({e}). Trying GT-style.")
# #                         # This is a fallback in case GAT now follows the GT constructor signature
# #                         if args.model_name == 'gat':
# #                              model = ModelClass(**gt_args)
# #                         else:
# #                              raise e # Re-raise if it wasn't GAT
# #                 else: # Graph Transformer
# #                      model = ModelClass(**gt_args)
# #                 # --- END MODIFIED SECTION ---

# #             except Exception as e:
# #                 print(f"ERROR: Failed to construct model for {param_string}. Skipping run. Error: {e}")
# #                 continue

# #             # --- Load Saved Model (uses experiment_path) ---
# #             out_dir = os.path.join(experiment_path, f'run_{run}') # Path for this specific run
# #             model_path = os.path.join(out_dir, 'model.pt')

# #             if not os.path.exists(model_path):
# #                 print(f"ERROR: Model file not found at {model_path}")
# #                 print(f"Skipping Run {run}.")
# #                 continue
                
# #             print(f"Loading model from: {model_path}")
# #             model.load_state_dict(torch.load(model_path, map_location=device))
# #             model.to(device)
# #             model.eval() 
# #             # --- END MODIFIED SECTION ---

# #             # --- Final Evaluation (Identical to original script) ---
# #             y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
# #             y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

# #             report_optimal = None
# #             ranking_metrics_run = {}

# #             if is_binary:
# #                 if y_valid_pred.shape[1] == 1:
# #                     valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy()
# #                 else:
# #                     valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
                
# #                 valid_true = y_valid_true.cpu().numpy().squeeze()
# #                 best_f1 = -1
# #                 best_threshold = 0.5
# #                 for threshold in np.linspace(0, 1, 100):
# #                     f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
# #                     if f1 > best_f1:
# #                         best_f1, best_threshold = f1, threshold
                
# #                 if y_test_pred.shape[1] == 1:
# #                     test_probs_np = torch.sigmoid(y_test_pred).squeeze().cpu().numpy()
# #                 else:
# #                     test_probs_np = torch.softmax(y_test_pred, dim=1)[:, 1].cpu().numpy()

# #                 y_test_true_np = y_test_true.cpu().numpy().squeeze()
                
# #                 # --- Calculate Ranking Metrics ---
# #                 ranking_metrics_run = calculate_ranking_metrics(y_test_true_np, test_probs_np, k_percentages)
# #                 print(f"Run {run} Ranking Metrics (Top 1%): P={ranking_metrics_run.get('precision_at_1_pct', 0):.4f}, R={ranking_metrics_run.get('recall_at_1_pct', 0):.4f}")

# #                 y_pred_optimal_np = (test_probs_np >= best_threshold).astype(int)
                
# #                 test_acc_opt = accuracy_score(y_test_true_np, y_pred_optimal_np)
# #                 test_bal_acc_opt = balanced_accuracy_score(y_test_true_np, y_pred_optimal_np)
# #                 report_optimal = classification_report(y_test_true_np, y_pred_optimal_np, output_dict=True, zero_division=0)
                
# #                 test_metrics_default['acc'] = test_acc_opt
# #                 test_metrics_default['balacc'] = test_bal_acc_opt

# #             # --- Aggregating and Saving Run Results ---
# #             acc = test_metrics_default['acc']
# #             bal_acc = test_metrics_default['balacc']
# #             roc_auc = test_metrics_default['rocauc']
# #             pr_auc = test_metrics_default.get('prauc', test_metrics_default.get('ap', 0))
            
# #             for key, value in ranking_metrics_run.items():
# #                 metric_base = key.split('_at_')[0]
# #                 k_val = key.split('_at_')[1]
# #                 full_key = f"{metric_base}_at_{k_val}"

# #                 if full_key in all_ranking_metrics:
# #                     all_ranking_metrics[full_key].append(value)

# #             y_pred_default_np = torch.argmax(y_test_pred, dim=1).cpu().numpy() if y_test_pred.shape[1] > 1 else (y_test_pred.cpu().numpy() > 0).astype(int)
# #             y_test_true_np = y_test_true.cpu().numpy()
# #             report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
            
# #             all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
# #             all_reports_default.append(report_default)
# #             if report_optimal: all_reports_optimal.append(report_optimal)

# #             # --- We still update the per-run metrics.json in the run_{run} folder ---
# #             os.makedirs(out_dir, exist_ok=True)
            
# #             run_metrics = {'accuracy': acc, 'balanced_accuracy': bal_acc, 
# #                              'auc_roc': roc_auc, 'auc_pr': pr_auc,
# #                              'classification_report_default': report_default, 
# #                              'classification_report_optimal': report_optimal}
            
# #             run_metrics.update(ranking_metrics_run) # Add the new ranking metrics

# #             # Try to load best_validation_metric from existing file
# #             try:
# #                 with open(os.path.join(out_dir, 'metrics.json'), 'r') as f:
# #                     existing_metrics = json.load(f)
# #                     if 'best_validation_metric' in existing_metrics:
# #                         run_metrics['best_validation_metric'] = existing_metrics['best_validation_metric']
# #                         all_best_valid.append(existing_metrics['best_validation_metric'])
# #             except FileNotFoundError:
# #                 pass # No existing file

# #             # Overwrite the metrics.json file with all new data
# #             with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
# #                 json.dump(run_metrics, f, indent=2)
        
# #         # --- End of inner 'run' loop ---

# #         # --- Final Summary (for this experiment) ---
        
# #         # --- Save summary to 'topk_percent' folder *inside this experiment_path* ---
# #         summary_dir = os.path.join(experiment_path, 'topk_percent')
# #         os.makedirs(summary_dir, exist_ok=True)
# #         print(f"\nSaving final summary and plots for {param_string} to: {summary_dir}")
        
# #         # --- Create summary with ONLY ranking metrics ---
# #         summary = {}

# #         # Add ranking metrics to final summary
# #         for key, values in all_ranking_metrics.items():
# #             if values: # Only add if we have data
# #                 summary[f'{key}_mean'] = np.mean(values)
# #                 summary[f'{key}_std'] = np.std(values)

# #         with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
# #             json.dump(summary, f, indent=2)
        
# #         # --- Call the plotting function ---
# #         save_ranking_plots(summary, k_percentages, summary_dir)

# #         print(f"\n{'='*40}\nFINAL SUMMARY (Top K%) for {param_string}\n{'='*40}")
# #         print(json.dumps(summary, f, indent=2))
    
# #     # --- End of outer 'experiment' loop ---
# #     print(f"\n{'='*50}\nAll experiments evaluated.\n{'='*50}")


# # if __name__ == "__main__":
# #     main()
# # # ```

# # # ### How to Run:

# # # Now, you just run one command. For example:

# # # ```bash
# # # python eval_only.py --dataset ogbg-molhiv --model_name gcn --runs 3