#!/usr/bin/env python3
import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import ndcg_score
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
from torch_geometric.nn import GPSConv, global_mean_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing

# =====================================================================================
# GatedGCNConv (unchanged)
# =====================================================================================

class GatedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr="add")
        self.A, self.B, self.C, self.D = [nn.Linear(in_channels, out_channels) for _ in range(4)]
        self.E = nn.Linear(edge_dim, out_channels)
        self.bn_node = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.E.in_features),
                                    device=x.device, dtype=x.dtype)
        edge_attr = edge_attr.float()

        Ax, Bx, Cx, Dx = self.A(x), self.B(x), self.C(x), self.D(x)

        orig_edges = edge_index.size(1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr.size(0) != edge_index.size(1):
            edge_attr = torch.cat(
                [edge_attr,
                 torch.zeros(edge_index.size(1) - orig_edges, edge_attr.size(1),
                             device=x.device, dtype=x.dtype)],
                dim=0
            )

        Ee = self.E(edge_attr)
        out = self.propagate(edge_index, Ax=Ax, Bx=Bx, Cx=Cx, Dx=Dx, Ee=Ee)
        return F.relu(self.bn_node(out))

    def message(self, Ax_i, Bx_j, Cx_i, Dx_j, Ee):
        return Ax_i + torch.sigmoid(Cx_i + Dx_j + Ee) * Bx_j


# =====================================================================================
# GPS MODEL (eval-safe)
# =====================================================================================

class GPS(nn.Module):
    def __init__(self, in_dim, channels, pe_dim, num_layers, edge_dim, pool):
        super().__init__()
        self.pe_dim = pe_dim
        self.node_emb = nn.Linear(in_dim, channels - pe_dim)
        self.pe_lin = nn.Linear(pe_dim, pe_dim) if pe_dim > 0 else None

        self.convs = nn.ModuleList([
            GPSConv(channels, GatedGCNConv(channels, channels, edge_dim), heads=4)
            for _ in range(num_layers)
        ])

        self.pool = pool
        self.lin_out = nn.Linear(channels, 1)

    def forward(self, x, x_pe, edge_index, edge_attr, batch):
        if x is None:
            x = torch.ones((edge_index.max().item() + 1, 1), device=edge_index.device)
        if x.dim() == 1:
            x = x.view(-1, 1)

        x = self.node_emb(x.float())

        if self.pe_dim > 0:
            if x_pe is None:
                x_pe = torch.zeros(x.size(0), self.pe_dim, device=x.device)
            x = torch.cat([x, self.pe_lin(x_pe.float())], dim=1)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_mean_pool(x, batch) if self.pool == "mean" else global_add_pool(x, batch)
        return self.lin_out(x)


# =====================================================================================
# RWSE
# =====================================================================================

def compute_rw_diag(data, steps):
    N = data.num_nodes
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=N).astype(float)
    deg = A.sum(1).A1
    P = A.multiply(1.0 / np.maximum(deg, 1e-12)[:, None]).toarray()

    cur = np.eye(N)
    diags = [np.diag(cur)]
    for _ in range(steps):
        cur = cur @ P
        diags.append(np.diag(cur))

    return torch.tensor(np.stack(diags, axis=1), dtype=torch.float)


def compute_projected_rwse(data, rw_steps, pe_dim):
    proj = nn.Linear(rw_steps + 1, pe_dim, bias=False)
    nn.init.orthogonal_(proj.weight)
    proj.eval()
    with torch.no_grad():
        return proj(compute_rw_diag(data, rw_steps))


def infer_pe_dim(state_dict, channels):
    return channels - state_dict["node_emb.weight"].size(0)


# =====================================================================================
# METRICS
# =====================================================================================

K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))

def calculate_ranking_metrics(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    total_pos = y_true.sum()
    n = len(y_true)

    metrics = {}
    for k_pct in K_PERCENTAGES:
        k = max(1, min(int(np.ceil(n * k_pct / 100)), n))
        pos_k = y_true[:k].sum()

        metrics[f"precision_at_{k_pct}_pct"] = float(pos_k / k)
        metrics[f"recall_at_{k_pct}_pct"] = float(pos_k / total_pos) if total_pos > 0 else 0.0
        metrics[f"ndcg_at_{k_pct}_pct"] = float(
            ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)
        )

    return metrics


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rw_steps", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    dataset = TUDataset(root="data/TUDataset", name=args.dataset)
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    test_dataset = dataset[idx[: len(dataset) // 10]]

    base_path = os.path.join("results_weighted", args.dataset, "GPS")
    folders = [f for f in os.listdir(base_path) if "pool" in f]

    for folder in folders:
        print(f"\nProcessing {folder}")
        exp_path = os.path.join(base_path, folder)
        out_dir = os.path.join(exp_path, "topk_results")
        os.makedirs(out_dir, exist_ok=True)

        parts = folder.split("_")
        channels = int([p[2:] for p in parts if p.startswith("ch")][0])
        num_layers = int([p[6:] for p in parts if p.startswith("layers")][0])
        pool = [p[4:] for p in parts if p.startswith("pool")][0]

        all_metrics = {f"{m}_at_{k}_pct": [] for k in K_PERCENTAGES for m in ["precision", "recall", "ndcg"]}

        for run in range(1, args.runs + 1):
            model_path = os.path.join(exp_path, f"run_{run}", "best_model.pt")
            if not os.path.exists(model_path):
                continue

            state_dict = torch.load(model_path, map_location="cpu")
            pe_dim = infer_pe_dim(state_dict, channels)

            for d in test_dataset:
                d.pe = compute_projected_rwse(d, args.rw_steps, pe_dim)
                if d.x is None:
                    d.x = torch.ones((d.num_nodes, 1))

            loader = DataLoader(test_dataset, batch_size=32)

            model = GPS(
                in_dim=dataset.num_node_features or 1,
                channels=channels,
                pe_dim=pe_dim,
                num_layers=num_layers,
                edge_dim=0,
                pool=pool,
            ).to(device)

            model.load_state_dict(state_dict)
            model.eval()

            y_true, y_score = [], []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data.x, getattr(data, "pe", None), data.edge_index, data.edge_attr, data.batch)
                    y_true.append(data.y.view(-1).cpu())
                    y_score.append(torch.sigmoid(out.view(-1)).cpu())

            metrics = calculate_ranking_metrics(
                torch.cat(y_true).numpy(),
                torch.cat(y_score).numpy(),
            )

            with open(os.path.join(out_dir, f"run_{run}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            for k, v in metrics.items():
                all_metrics[k].append(v)

        summary = {}
        for k, v in all_metrics.items():
            if v:
                summary[f"{k}_mean"] = float(np.mean(v))
                summary[f"{k}_std"] = float(np.std(v))

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)


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
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
# from torch_geometric.transforms import OneHotDegree
# from torch_geometric.utils import degree, to_scipy_sparse_matrix
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
# from tqdm import tqdm

# # --- Import GPS Specific Components from your training logic ---
# from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool, global_add_pool
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import add_self_loops

# # =====================================================================================

# #   GPS Model Architecture
# # =====================================================================================

# class GatedGCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dim):
#         super().__init__(aggr="add")
#         self.in_channels, self.out_channels, self.edge_dim = in_channels, out_channels, edge_dim
#         self.A, self.B, self.C, self.D = [nn.Linear(in_channels, out_channels) for _ in range(4)]
#         self.E = nn.Linear(edge_dim, out_channels)
#         self.bn_node = nn.BatchNorm1d(out_channels)

#     def forward(self, x, edge_index, edge_attr=None):
#         if edge_attr is None:
#             edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device, dtype=x.dtype)
#         edge_attr = edge_attr.float()
#         Ax, Bx, Cx, Dx = self.A(x), self.B(x), self.C(x), self.D(x)
#         orig_edges = edge_index.size(1)
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#         if edge_attr.size(0) != edge_index.size(1):
#             n_new = edge_index.size(1) - orig_edges
#             edge_attr = torch.cat([edge_attr, torch.zeros((n_new, edge_attr.size(1)), device=x.device, dtype=x.dtype)], dim=0)
#         Ee = self.E(edge_attr)
#         out = self.propagate(edge_index, Ax=Ax, Bx=Bx, Cx=Cx, Dx=Dx, Ee=Ee)
#         out = self.bn_node(out)
#         return F.relu(out)

#     def message(self, Ax_i, Bx_j, Cx_i, Dx_j, Ee):
#         gate = torch.sigmoid(Cx_i + Dx_j + Ee)
#         return Ax_i + gate * Bx_j

# class GPS(nn.Module):
#     def __init__(self, in_dim, channels, pe_dim, num_layers, conv_type="gated", edge_dim=0, pool="mean"):
#         super().__init__()
#         node_out_dim = channels - pe_dim if pe_dim > 0 else channels
#         self.node_emb = nn.Linear(in_dim, node_out_dim)
#         self.pe_lin = nn.Linear(pe_dim, pe_dim) if pe_dim > 0 else None
#         self.convs = nn.ModuleList()
#         self.pool = pool
#         for _ in range(num_layers):
#             if conv_type == "gated":
#                 mpnn = GatedGCNConv(channels, channels, edge_dim=edge_dim)
#                 layer = GPSConv(channels, mpnn, heads=4)
#             else:
#                 nn_seq = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))
#                 layer = GPSConv(channels, GINEConv(nn_seq, edge_dim=edge_dim), heads=4, attn_dropout=0.5)
#             self.convs.append(layer)
#         self.lin_out = nn.Linear(channels, 1)

#     def forward(self, x, x_pe, edge_index, edge_attr, batch):
#         if x.dim() == 1: x = x.view(-1, 1)
#         x = x.float()
#         if self.pe_lin is not None and x_pe is not None:
#             x = torch.cat([self.node_emb(x), self.pe_lin(x_pe.float())], dim=1)
#         else:
#             x = self.node_emb(x)
#         for conv in self.convs:
#             x = conv(x, edge_index, batch, edge_attr=edge_attr)
#         x = global_mean_pool(x, batch) if self.pool == "mean" else global_add_pool(x, batch)
#         return self.lin_out(x)

# # =====================================================================================
# #   Helpers
# # =====================================================================================

# def compute_rw_diag(data, steps: int):
#     N = data.num_nodes
#     if N == 0: return torch.zeros((0, steps+1), dtype=torch.float)
#     A = to_scipy_sparse_matrix(data.edge_index, num_nodes=N).astype(float)
#     deg = A.sum(1).A1
#     D_inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg!=0)
#     P = A.multiply(D_inv[:,None]).toarray()
#     cur = np.eye(N, dtype=float)
#     diags = [np.diag(cur).copy()]
#     for _ in range(steps):
#         cur = cur @ P
#         diags.append(np.diag(cur).copy())
#     return torch.tensor(np.stack(diags, axis=1), dtype=torch.float)

# def parse_gps_folder(folder_name, base_args):
#     """Robustly parses parameters from folder name."""
#     args = argparse.Namespace(**vars(base_args))
#     parts = folder_name.split('_')
#     for p in parts:
#         try:
#             if p.startswith('ch'): args.channels = int(p[2:])
#             elif p.startswith('rwse'): args.rwse_dim = int(p[4:])
#             elif p.startswith('layers'): args.num_layers = int(p[6:])
#             elif p.startswith('conv'): args.conv_type = p[4:]
#             elif p.startswith('pool'): args.pool = p[4:]
#         except (ValueError, IndexError):
#             continue
#     return args

# # =====================================================================================
# #   Evaluation Logic
# # =====================================================================================

# K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))

# def calculate_ranking_metrics(y_true, y_score, k_percentages):
#     y_true, y_score = np.asarray(y_true).squeeze(), np.asarray(y_score).squeeze()
#     sort_indices = np.argsort(y_score)[::-1]
#     y_true_sorted = y_true[sort_indices]
#     total_positives = np.sum(y_true)
#     total_samples = len(y_true_sorted)
#     metrics = {}
#     for k_pct in k_percentages:
#         k = max(1, min(int(np.ceil(total_samples * (k_pct / 100.0))), total_samples))
#         num_pos_in_k = np.sum(y_true_sorted[:k])
#         m_key = f"at_{k_pct}_pct"
#         metrics[f'precision_{m_key}'] = float(num_pos_in_k / k)
#         metrics[f'recall_{m_key}'] = float(num_pos_in_k / total_positives) if total_positives > 0 else 0.0
#         metrics[f'ndcg_{m_key}'] = float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k))
#     return metrics

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, required=True)
#     parser.add_argument('--runs', type=int, default=3)
#     parser.add_argument('--filter', type=str, default=None)
#     parser.add_argument('--rw_steps', type=int, default=16)
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--seed', type=int, default=42)
#     base_args = parser.parse_args()

#     device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
    
#     # 1. Load Data
#     print(f"--- Loading {base_args.dataset} ---")
#     if 'ogb' in base_args.dataset:
#         dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
#         test_idx = dataset.get_idx_split()["test"]
#         test_dataset = dataset[test_idx]
#     else:
#         dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
#         indices = list(range(len(dataset)))
#         random.seed(base_args.seed); random.shuffle(indices)
#         test_dataset = dataset[indices[:len(dataset)//10]]

#     print("Computing RWSE for test set...")
#     for data in tqdm(test_dataset):
#         data.pe = compute_rw_diag(data, steps=base_args.rw_steps)
#         if data.x is None: data.x = torch.ones((data.num_nodes, 1))
    
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     # 2. Iterate Experiments
#     base_path = os.path.join("results_weighted", base_args.dataset, "GPS")
#     if not os.path.exists(base_path): return

#     # FIX: Skip hidden folders and only process folders following the training naming pattern
#     folders = [f for f in os.listdir(base_path) 
#                if os.path.isdir(os.path.join(base_path, f)) 
#                and not f.startswith('.') 
#                and 'pool' in f]

#     if base_args.filter: 
#         folders = [f for f in folders if base_args.filter in f]

#     for folder in folders:
#         if folder in ['summary', 'topk_results']: continue
#         print(f"Processing: {folder}")
#         exp_path = os.path.join(base_path, folder)
#         out_dir = os.path.join(exp_path, 'topk_results')
#         os.makedirs(out_dir, exist_ok=True)
        
#         args = parse_gps_folder(folder, base_args)
#         all_metrics = {f'{m}_at_{k}_pct': [] for k in K_PERCENTAGES for m in ['precision', 'recall', 'ndcg']}

#         for run in range(1, base_args.runs + 1):
#             model_path = os.path.join(exp_path, f'run_{run}', "best_model.pt")
#             if not os.path.exists(model_path): continue

#             # Initialize Model
#             in_dim = dataset.num_node_features if dataset.num_node_features > 0 else 1
#             edge_dim = dataset[0].edge_attr.size(-1) if dataset[0].edge_attr is not None else 0
            
#             # Use pe_dim = args.rwse_dim parsed from the folder name
#             model = GPS(in_dim=in_dim, channels=args.channels, pe_dim=args.rwse_dim,
#                         num_layers=args.num_layers, conv_type=args.conv_type, 
#                         edge_dim=edge_dim, pool=args.pool).to(device)
            
#             model.load_state_dict(torch.load(model_path, map_location=device))
#             model.eval()

#             # Inference
#             y_true_list, y_prob_list = [], []
#             with torch.no_grad():
#                 for data in test_loader:
#                     data = data.to(device)
#                     out = model(data.x, getattr(data, "pe", None), data.edge_index, data.edge_attr, data.batch)
#                     y_true_list.append(data.y.view(-1).cpu())
#                     y_prob_list.append(torch.sigmoid(out.view(-1)).cpu())
            
#             y_true, y_probs = torch.cat(y_true_list).numpy(), torch.cat(y_prob_list).numpy()
#             run_metrics = calculate_ranking_metrics(y_true, y_probs, K_PERCENTAGES)
#             for k, v in run_metrics.items(): all_metrics[k].append(v)
            
#             with open(os.path.join(out_dir, f'run_{run}_metrics.json'), 'w') as f:
#                 json.dump(run_metrics, f, indent=4)

#         # Summary
#         summary = {f'{k}_mean': float(np.mean(v)) for k, v in all_metrics.items() if v}
#         summary.update({f'{k}_std': float(np.std(v)) for k, v in all_metrics.items() if v})
#         with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
#             json.dump(summary, f, indent=4)

# if __name__ == "__main__":
#     main()





# # # import argparse
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # import numpy as np
# # # import os
# # # import json
# # # import random
# # # from sklearn.metrics import ndcg_score
# # # from torch_geometric.loader import DataLoader
# # # from torch_geometric.datasets import TUDataset
# # # from torch_geometric.transforms import OneHotDegree
# # # from torch_geometric.utils import degree, to_scipy_sparse_matrix
# # # from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
# # # from tqdm import tqdm

# # # # --- Import GPS Specific Components ---
# # # # Ensure these classes match your training script exactly
# # # from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool, global_add_pool
# # # from torch_geometric.nn.conv import MessagePassing
# # # from torch_geometric.utils import add_self_loops

# # # # =====================================================================================
# # # #   GPS Model Architecture (Extracted from your training code)
# # # # =====================================================================================

# # # class GatedGCNConv(MessagePassing):
# # #     def __init__(self, in_channels, out_channels, edge_dim):
# # #         super().__init__(aggr="add")
# # #         self.in_channels, self.out_channels, self.edge_dim = in_channels, out_channels, edge_dim
# # #         self.A, self.B, self.C, self.D = [nn.Linear(in_channels, out_channels) for _ in range(4)]
# # #         self.E = nn.Linear(edge_dim, out_channels)
# # #         self.bn_node = nn.BatchNorm1d(out_channels)

# # #     def forward(self, x, edge_index, edge_attr=None):
# # #         if edge_attr is None:
# # #             edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device, dtype=x.dtype)
# # #         edge_attr = edge_attr.float()
# # #         Ax, Bx, Cx, Dx = self.A(x), self.B(x), self.C(x), self.D(x)
# # #         orig_edges = edge_index.size(1)
# # #         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
# # #         if edge_attr.size(0) != edge_index.size(1):
# # #             n_new = edge_index.size(1) - orig_edges
# # #             edge_attr = torch.cat([edge_attr, torch.zeros((n_new, edge_attr.size(1)), device=x.device, dtype=x.dtype)], dim=0)
# # #         Ee = self.E(edge_attr)
# # #         out = self.propagate(edge_index, Ax=Ax, Bx=Bx, Cx=Cx, Dx=Dx, Ee=Ee)
# # #         out = self.bn_node(out)
# # #         return F.relu(out)

# # #     def message(self, Ax_i, Bx_j, Cx_i, Dx_j, Ee):
# # #         gate = torch.sigmoid(Cx_i + Dx_j + Ee)
# # #         return Ax_i + gate * Bx_j

# # # class GPS(nn.Module):
# # #     def __init__(self, in_dim, channels, pe_dim, num_layers, conv_type="gated", edge_dim=0, pool="mean"):
# # #         super().__init__()
# # #         node_out_dim = channels - pe_dim if pe_dim > 0 else channels
# # #         self.node_emb = nn.Linear(in_dim, node_out_dim)
# # #         self.pe_lin = nn.Linear(pe_dim, pe_dim) if pe_dim > 0 else None
# # #         self.convs = nn.ModuleList()
# # #         self.pool = pool
# # #         for _ in range(num_layers):
# # #             if conv_type == "gated":
# # #                 mpnn = GatedGCNConv(channels, channels, edge_dim=edge_dim)
# # #                 layer = GPSConv(channels, mpnn, heads=4)
# # #             else:
# # #                 nn_seq = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))
# # #                 layer = GPSConv(channels, GINEConv(nn_seq, edge_dim=edge_dim), heads=4, attn_dropout=0.5)
# # #             self.convs.append(layer)
# # #         self.lin_out = nn.Linear(channels, 1)

# # #     def forward(self, x, x_pe, edge_index, edge_attr, batch):
# # #         if x.dim() == 1: x = x.view(-1, 1)
# # #         x = x.float()
# # #         if self.pe_lin is not None and x_pe is not None:
# # #             x = torch.cat([self.node_emb(x), self.pe_lin(x_pe.float())], dim=1)
# # #         else:
# # #             x = self.node_emb(x)
# # #         for conv in self.convs:
# # #             x = conv(x, edge_index, batch, edge_attr=edge_attr)
# # #         x = global_mean_pool(x, batch) if self.pool == "mean" else global_add_pool(x, batch)
# # #         return self.lin_out(x)

# # # # =====================================================================================
# # # #   Helper: RWSE Computation
# # # # =====================================================================================

# # # def compute_rw_diag(data, steps: int):
# # #     N = data.num_nodes
# # #     if N == 0: return torch.zeros((0, steps+1), dtype=torch.float)
# # #     A = to_scipy_sparse_matrix(data.edge_index, num_nodes=N).astype(float)
# # #     deg = A.sum(1).A1
# # #     D_inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg!=0)
# # #     P = A.multiply(D_inv[:,None]).toarray()
# # #     cur = np.eye(N, dtype=float)
# # #     diags = [np.diag(cur).copy()]
# # #     for _ in range(steps):
# # #         cur = cur @ P
# # #         diags.append(np.diag(cur).copy())
# # #     return torch.tensor(np.stack(diags, axis=1), dtype=torch.float)

# # # # =====================================================================================
# # # #   Metrics & Inference
# # # # =====================================================================================

# # # K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))

# # # def calculate_ranking_metrics(y_true, y_score, k_percentages):
# # #     y_true, y_score = np.asarray(y_true).squeeze(), np.asarray(y_score).squeeze()
# # #     sort_indices = np.argsort(y_score)[::-1]
# # #     y_true_sorted = y_true[sort_indices]
# # #     total_positives = np.sum(y_true)
# # #     total_samples = len(y_true_sorted)
# # #     metrics = {}
# # #     for k_pct in k_percentages:
# # #         k = max(1, min(int(np.ceil(total_samples * (k_pct / 100.0))), total_samples))
# # #         num_pos_in_k = np.sum(y_true_sorted[:k])
# # #         m_key = f"at_{k_pct}_pct"
# # #         metrics[f'precision_{m_key}'] = float(num_pos_in_k / k)
# # #         metrics[f'recall_{m_key}'] = float(num_pos_in_k / total_positives) if total_positives > 0 else 0.0
# # #         metrics[f'ndcg_{m_key}'] = float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k))
# # #     return metrics

# # # def parse_gps_folder(folder_name, base_args):
# # #     """Extends args from the pool_mean_metricrocauc... naming convention"""
# # #     args = argparse.Namespace(**vars(base_args))
# # #     parts = folder_name.split('_')
# # #     for p in parts:
# # #         if p.startswith('ch'): args.channels = int(p[2:])
# # #         if p.startswith('rwse'): args.rwse_dim = int(p[4:])
# # #         if p.startswith('layers'): args.num_layers = int(p[6:])
# # #         if p.startswith('conv'): args.conv_type = p[4:]
# # #         if p.startswith('pool'): args.pool = p[4:]
# # #     return args

# # # # =====================================================================================
# # # #   Main
# # # # =====================================================================================

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument('--dataset', type=str, required=True)
# # #     parser.add_argument('--runs', type=int, default=3)
# # #     parser.add_argument('--filter', type=str, default=None)
# # #     parser.add_argument('--rw_steps', type=int, default=16) # Should match training
# # #     parser.add_argument('--device', type=int, default=0)
# # #     parser.add_argument('--seed', type=int, default=42)
# # #     base_args = parser.parse_args()

# # #     device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
    
# # #     # 1. Load Data
# # #     print(f"--- Loading {base_args.dataset} ---")
# # #     if 'ogb' in base_args.dataset:
# # #         dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
# # #         test_idx = dataset.get_idx_split()["test"]
# # #         test_dataset = dataset[test_idx]
# # #     else:
# # #         dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
# # #         indices = list(range(len(dataset)))
# # #         random.seed(base_args.seed); random.shuffle(indices)
# # #         test_dataset = dataset[indices[:len(dataset)//10]]

# # #     # Compute RWSE for evaluation
# # #     print("Computing RWSE for test set...")
# # #     for data in tqdm(test_dataset):
# # #         data.pe = compute_rw_diag(data, steps=base_args.rw_steps)
# # #         if data.x is None: data.x = torch.ones((data.num_nodes, 1))
    
# # #     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # #     # 2. Iterate Experiments
# # #     base_path = os.path.join("results_weighted", base_args.dataset, "GPS")
# # #     if not os.path.exists(base_path): return

# # #     folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
# # #     if base_args.filter: folders = [f for f in folders if base_args.filter in f]

# # #     for folder in folders:
# # #         if folder in ['summary', 'topk_results']: continue
# # #         print(f"Processing: {folder}")
# # #         exp_path = os.path.join(base_path, folder)
# # #         out_dir = os.path.join(exp_path, 'topk_results')
# # #         os.makedirs(out_dir, exist_ok=True)
        
# # #         args = parse_gps_folder(folder, base_args)
# # #         all_metrics = {f'{m}_at_{k}_pct': [] for k in K_PERCENTAGES for m in ['precision', 'recall', 'ndcg']}

# # #         for run in range(1, base_args.runs + 1):
# # #             model_path = os.path.join(exp_path, f'run_{run}', "best_model.pt")
# # #             if not os.path.exists(model_path): continue

# # #             # Initialize Model
# # #             in_dim = dataset.num_node_features if dataset.num_node_features > 0 else 1
# # #             edge_dim = dataset[0].edge_attr.size(-1) if dataset[0].edge_attr is not None else 0
            
# # #             # Note: pe_dim in training was rwse_dim (which was a projection of rw_steps + 1)
# # #             # We use the rwse_dim parsed from the folder name
# # #             model = GPS(in_dim=in_dim, channels=args.channels, pe_dim=args.rwse_dim,
# # #                         num_layers=args.num_layers, conv_type=args.conv_type, 
# # #                         edge_dim=edge_dim, pool=args.pool).to(device)
            
# # #             model.load_state_dict(torch.load(model_path, map_location=device))
# # #             model.eval()

# # #             # Inference
# # #             y_true_list, y_prob_list = [], []
# # #             with torch.no_grad():
# # #                 for data in test_loader:
# # #                     data = data.to(device)
# # #                     out = model(data.x, getattr(data, "pe", None), data.edge_index, data.edge_attr, data.batch)
# # #                     y_true_list.append(data.y.view(-1).cpu())
# # #                     y_prob_list.append(torch.sigmoid(out.view(-1)).cpu())
            
# # #             y_true, y_probs = torch.cat(y_true_list).numpy(), torch.cat(y_prob_list).numpy()
            
# # #             # Metrics
# # #             run_metrics = calculate_ranking_metrics(y_true, y_probs, K_PERCENTAGES)
# # #             for k, v in run_metrics.items(): all_metrics[k].append(v)
            
# # #             with open(os.path.join(out_dir, f'run_{run}_metrics.json'), 'w') as f:
# # #                 json.dump(run_metrics, f, indent=4)

# # #         # Summary
# # #         summary = {f'{k}_mean': float(np.mean(v)) for k, v in all_metrics.items() if v}
# # #         summary.update({f'{k}_std': float(np.std(v)) for k, v in all_metrics.items() if v})
# # #         with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
# # #             json.dump(summary, f, indent=4)

# # # if __name__ == "__main__":
# # #     main()