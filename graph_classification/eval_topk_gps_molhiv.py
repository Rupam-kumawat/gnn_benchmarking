import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
import glob
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool, global_add_pool
from ogb.graphproppred import PygGraphPropPredDataset

# --- Matplotlib for Headless Plotting ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

# =====================================================================================
#   PART 1: MODEL DEFINITIONS (Must match training script exactly)
# =====================================================================================

class GatedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr="add")
        self.in_channels, self.out_channels, self.edge_dim = in_channels, out_channels, edge_dim
        self.A, self.B, self.C, self.D = [nn.Linear(in_channels, out_channels) for _ in range(4)]
        self.E = nn.Linear(edge_dim, out_channels)
        self.bn_node = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device, dtype=x.dtype)
        edge_attr = edge_attr.float()
        Ax, Bx, Cx, Dx = self.A(x), self.B(x), self.C(x), self.D(x)
        orig_edges = edge_index.size(1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr.size(0) != edge_index.size(1):
            n_new = edge_index.size(1) - orig_edges
            edge_attr = torch.cat([edge_attr, torch.zeros((n_new, edge_attr.size(1)), device=x.device, dtype=x.dtype)], dim=0)
        Ee = self.E(edge_attr)
        out = self.propagate(edge_index, Ax=Ax, Bx=Bx, Cx=Cx, Dx=Dx, Ee=Ee)
        out = self.bn_node(out)
        return F.relu(out)

    def message(self, Ax_i, Bx_j, Cx_i, Dx_j, Ee):
        gate = torch.sigmoid(Cx_i + Dx_j + Ee)
        return Ax_i + gate * Bx_j

class GPS(nn.Module):
    def __init__(self, in_dim, channels, pe_dim, num_layers, conv_type="gated", edge_dim=0, pool="mean"):
        super().__init__()
        node_out_dim = channels - pe_dim if pe_dim > 0 else channels
        if node_out_dim <= 0: raise ValueError("channels must be > pe_dim")
        self.node_emb = nn.Linear(in_dim, node_out_dim)
        self.pe_lin = nn.Linear(pe_dim, pe_dim) if pe_dim>0 else None
        self.convs = nn.ModuleList()
        self.pool=pool
        for _ in range(num_layers):
            if conv_type=="gated":
                mpnn = GatedGCNConv(channels, channels, edge_dim=edge_dim)
                layer = GPSConv(channels, mpnn, heads=4)
            else:
                nn_seq = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))
                layer = GPSConv(channels, GINEConv(nn_seq, edge_dim=edge_dim), heads=4, attn_dropout=0.5)
            self.convs.append(layer)
        self.lin_out = nn.Linear(channels, 1)

    def forward(self, x, x_pe, edge_index, edge_attr, batch):
        x = x if x is not None else torch.ones((edge_index.max().item()+1, 1), device=edge_index.device)
        if x.dim()==1: x=x.view(-1,1)
        x = x.float()
        if self.pe_lin is not None and x_pe is not None:
            x = torch.cat([self.node_emb(x), self.pe_lin(x_pe.float())], dim=1)
        else:
            x = self.node_emb(x)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        if self.pool=="mean":
            x = global_mean_pool(x, batch)
        elif self.pool=="sum":
            x = global_add_pool(x, batch)
        return self.lin_out(x)

def compute_rw_diag(data, steps: int):
    N = data.num_nodes
    if N == 0: return torch.zeros((0, steps+1), dtype=torch.float)
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=N).astype(float)
    deg = A.sum(1).A1
    D_inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg!=0)
    P = A.multiply(D_inv[:,None]).toarray()
    cur = np.eye(N, dtype=float)
    diags = [np.diag(cur).copy()]
    for _ in range(steps):
        cur = cur @ P
        diags.append(np.diag(cur).copy())
    return torch.tensor(np.stack(diags, axis=1), dtype=torch.float)

# =====================================================================================
#   PART 2: RANKING METRICS & PLOTTING (Identical to previous logic)
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
        metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives if total_positives > 0 else 0.0
        
        true_relevance = y_true.reshape(1, -1)
        pred_scores = y_score.reshape(1, -1)
        metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

    return metrics

def save_ranking_plots(summary_data, k_percentages, output_dir):
    metric_bases = ['precision', 'recall', 'ndcg']
    for metric_base in metric_bases:
        try:
            means = [summary_data.get(f'{metric_base}_at_{k}_pct_mean', np.nan) for k in k_percentages]
            stds = [summary_data.get(f'{metric_base}_at_{k}_pct_std', np.nan) for k in k_percentages]
            means = np.array(means); stds = np.array(stds)
            valid_indices = ~np.isnan(means)
            if not np.any(valid_indices): continue

            x_axis = np.array(k_percentages)[valid_indices]
            plot_means = means[valid_indices]
            plot_stds = stds[valid_indices]

            plt.figure(figsize=(10, 6))
            plt.plot(x_axis, plot_means, 'o-', label='Mean')
            plt.fill_between(x_axis, plot_means - plot_stds, plot_means + plot_stds, color='blue', alpha=0.2, label='Mean ± 1 Std Dev')
            plt.title(f'{metric_base.capitalize()} vs. Top K%')
            plt.xlabel('Top K (%)')
            plt.ylabel(f'Mean {metric_base.capitalize()}')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f'{metric_base}_vs_k_percent.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate plot for {metric_base}. Error: {e}")

# =====================================================================================
#   PART 3: PDF GENERATION (Top=PRAUC, Bottom=ROCAUC)
# =====================================================================================

def generate_comparison_pdf(base_path, output_pdf_name):
    print(f"\n{'='*50}\nGenerating Comparison PDF: {output_pdf_name}\n{'='*50}")
    
    experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    experiment_dirs = [d for d in experiment_dirs if d not in ['summary', 'topk_percent'] and not d.startswith('.')]

    # Group experiments by everything EXCEPT the metric
    grouped_experiments = {}

    for folder_name in experiment_dirs:
        # Custom parsing for the specific GPS folder string format:
        # pool_mean_metricrocauc_convgated_...
        try:
            parts = folder_name.split('_')
            params = {}
            metric = None
            
            # Extract key-values. 
            # Note: The format is keyvalue (e.g., metricrocauc) not key-value in some places?
            # Based on your training script: 
            # f"pool_{args.pool}_metric{args.metric}_conv{args.conv_type}_layers{args.num_layers}..."
            
            # We need to reconstruct the config key excluding the metric part.
            # Simple approach: Find the part starting with 'metric' and extract the value.
            
            config_parts = []
            for part in parts:
                if part.startswith('metric'):
                    metric = part.replace('metric', '')
                else:
                    config_parts.append(part)
            
            if metric is None: continue

            config_key = tuple(sorted(config_parts))
            if config_key not in grouped_experiments: grouped_experiments[config_key] = {}
            grouped_experiments[config_key][metric] = os.path.join(base_path, folder_name)
        except:
            continue

    pdf_path = os.path.join(base_path, output_pdf_name)
    
    with PdfPages(pdf_path) as pdf:
        sorted_keys = sorted(grouped_experiments.keys())
        
        for config_key in sorted_keys:
            paths = grouped_experiments[config_key]
            path_prauc = paths.get('prauc')
            path_rocauc = paths.get('rocauc')
            
            if not path_prauc and not path_rocauc: continue

            config_str = " | ".join(config_key)
            print(f"Adding page for config")

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
                            ax.imshow(mpimg.imread(img_path))
                        else:
                            ax.text(0.5, 0.5, "Plot not found", ha='center', va='center')
                else:
                    for col_idx in range(3):
                        ax = axes[row_idx, col_idx]; ax.axis('off')
                        ax.text(0.5, 0.5, f"No data for {metric_label}", ha='center', va='center')

            plot_row(0, path_prauc, 'prauc')
            plot_row(1, path_rocauc, 'rocauc')

            plt.suptitle(f"GPS Configuration", fontsize=15, fontweight='bold', y=0.98)
            # Add subtitle with params
            fig.text(0.5, 0.965, config_str, ha='center', fontsize=8)
            
            fig.text(0.5, 0.94, "METRIC: PRAUC (Precision-Recall AUC)", ha='center', va='center', fontsize=14, fontweight='bold', color='darkblue', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            fig.text(0.5, 0.49, "METRIC: ROCAUC (Receiver Operating Characteristic AUC)", ha='center', va='center', fontsize=14, fontweight='bold', color='darkred', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93]) 
            plt.subplots_adjust(hspace=0.4) 
            pdf.savefig(fig)
            plt.close()
    print(f"PDF Generated: {pdf_path}")

# =====================================================================================
#   PART 4: MAIN EXECUTION
# =====================================================================================

def parse_gps_folder(folder_name):
    """
    Parses: pool_{}_metric{}_conv{}_layers{}_ch{}_rwse{}_lr{}_bs{}...
    Returns dict of args
    """
    args = {}
    
    # Explicitly handle the 'pool' argument which uses an underscore separator
    if "pool_mean" in folder_name:
        args['pool'] = "mean"
    elif "pool_sum" in folder_name:
        args['pool'] = "sum"
    else:
        args['pool'] = "mean" # Default fallback
        
    parts = folder_name.split('_')
    for part in parts:
        if part.startswith('conv'): 
            args['conv_type'] = part.replace('conv', '')
        elif part.startswith('layers'): 
            args['num_layers'] = int(part.replace('layers', ''))
        elif part.startswith('ch'): 
            args['channels'] = int(part.replace('ch', ''))
        elif part.startswith('rwse'): 
            # If rwse is explicitly 0 in folder, keep it 0
            val = part.replace('rwse', '')
            args['rwse_dim'] = int(val) if val else None
        # 'pool' is already handled above
            
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--rw_steps", type=int, default=16, help="Must match training setting")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset & Precompute RWSE (REQUIRED for GPS)
    print(f"Loading {args.dataset} and computing RWSE (Steps={args.rw_steps})...")
    dataset = PygGraphPropPredDataset(root=f"./data/OGB", name=args.dataset)
    split_idx = dataset.get_idx_split()
    test_idx = split_idx["test"]
    
    # Compute RWSE projection matrix (Just to initialize PE dim logic)
    in_rw_dim = args.rw_steps + 1
    
    # We only need to process TEST data for evaluation
    # To be safe, we process the specific indices
    test_list = []
    print("Computing RWSE for Test Set...")
    for i in tqdm(test_idx):
        data = dataset[i]
        diag = compute_rw_diag(data, steps=args.rw_steps)
        # We don't project here, we just store the diag. 
        # The projection happens inside the model or via a linear layer before the model?
        # WAIT: In training code: `data.pe = rw_proj(diag)`. 
        # The model expects `x_pe` to be size `rwse_dim`.
        # The projection layer `rw_proj` was NOT part of the model class in the training script,
        # it was a separate linear layer in the training loop.
        # This is tricky. The trained `rw_proj` layer was NOT saved in `model.pt` in the original script!
        # The original script saved `model.state_dict()`. `model` is the GPS class.
        # The `rw_proj` was defined outside.
        # **CRITICAL FIX**: If `rw_proj` wasn't saved, we can't reproduce the exact embeddings.
        # HOWEVER, usually GPS implementations include the projection in the model.
        # Looking at your training code: `rw_proj` is defined in `if __name__`.
        # It is NOT passed to the model.
        # BUT, `data.pe` is passed to `model(..., x_pe=getattr(data,"pe"), ...)`
        # AND `GPS` class has `self.pe_lin = nn.Linear(pe_dim, pe_dim)`.
        
        # Issue: The training script calculates `data.pe = rw_proj(diag)` OUTSIDE the model.
        # Unless `rw_proj` was saved, we are stuck.
        # BUT: Check `rw_proj` initialization in training: `nn.init.orthogonal_(rw_proj.weight)`. 
        # It's a random projection. If we fix the seed, we might recover it.
        
        # Assumption: We must replicate the random seed fixation to get the same `rw_proj`.
        data.diag = diag # Store raw diag for now
        test_list.append(data)

    # 2. Re-generate the RW Projection Layer (using fixed seed 42 from training script)
    # This is a hack because the training script didn't save the projection layer.
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    
    base_path = f"results_weighted/{args.dataset}/GPS"
    if not os.path.exists(base_path): return

    experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    experiment_dirs = [d for d in experiment_dirs if 'summary' not in d]

    k_percentages = [1] + list(range(5, 101, 5))

    for folder in experiment_dirs:
        print(f"\nEvaluating Folder: {folder}")
        folder_args = parse_gps_folder(folder)
        
        # Re-create the specific RW Projection for this config
        actual_rwse_dim = folder_args.get('rwse_dim', in_rw_dim) # Default if not in name
        if actual_rwse_dim is None: actual_rwse_dim = in_rw_dim

        # Reset seed to 42 to match training script initialization of rw_proj
        torch.manual_seed(42) 
        rw_proj = nn.Linear(in_rw_dim, actual_rwse_dim, bias=False)
        nn.init.orthogonal_(rw_proj.weight)
        rw_proj.eval()

        # Prepare Data for this config
        final_test_list = []
        for data in test_list:
            d = data.clone()
            with torch.no_grad():
                d.pe = rw_proj(d.diag) # Apply the deterministic random projection
            final_test_list.append(d)
        
        loader = DataLoader(final_test_list, batch_size=32, shuffle=False)

        # Determine Model Channels (Logic from training script)
        channels = folder_args['channels']
        if actual_rwse_dim >= channels: channels = actual_rwse_dim + 1
        
        # Setup Model
        in_dim = dataset.num_node_features
        edge_dim = dataset.num_edge_features
        
        all_rank_metrics = {f'{m}_at_{k}_pct': [] for k in k_percentages for m in ['precision', 'recall', 'ndcg']}
        
        # Loop Runs
        for run in range(1, args.runs + 1):
            model_path = os.path.join(base_path, folder, f"run_{run}", "best_model.pt")
            if not os.path.exists(model_path): continue
            
            try:
                model = GPS(in_dim=in_dim, channels=channels, pe_dim=actual_rwse_dim,
                            num_layers=folder_args['num_layers'], 
                            conv_type=folder_args['conv_type'], 
                            edge_dim=edge_dim, 
                            pool=folder_args['pool']).to(device)
                
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
            except Exception as e:
                print(f"Error loading model: {e}"); continue

            # Inference
            y_true_all, y_score_all = [], []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data.x, getattr(data,"pe",None), data.edge_index, data.edge_attr, data.batch)
                    y_true_all.append(data.y.cpu().numpy())
                    y_score_all.append(torch.sigmoid(out).cpu().numpy()) # Binary
            
            y_true = np.concatenate(y_true_all)
            y_score = np.concatenate(y_score_all)
            
            # Calculate Metrics
            metrics = calculate_ranking_metrics(y_true, y_score, k_percentages)
            for k, v in metrics.items(): all_rank_metrics[k].append(v)

        # Save Plots
        summary_dir = os.path.join(base_path, folder, 'topk_percent')
        os.makedirs(summary_dir, exist_ok=True)
        summary = {}
        for key, values in all_rank_metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
        
        with open(os.path.join(summary_dir, 'summary.json'), 'w') as f: json.dump(summary, f, indent=2)
        save_ranking_plots(summary, k_percentages, summary_dir)

    # PDF Report
    generate_comparison_pdf(base_path, f"{args.dataset}_GPS_comparison.pdf")

if __name__ == "__main__":
    main()