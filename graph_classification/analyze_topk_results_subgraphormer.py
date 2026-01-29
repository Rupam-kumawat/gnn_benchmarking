#!/usr/bin/env python3
import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import ndcg_score
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset

# -------------------------
# Training-time imports
# -------------------------
from parse_graphclass import parser_add_main_args
from subgraphormer_model import Subgraphormer
from subgraphormer_utils import get_subgraphormer_transform

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# =====================================================================================
# Top-K configuration (SAME AS GPS)
# =====================================================================================

K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))


# =====================================================================================
# Ranking metrics
# =====================================================================================

def calculate_ranking_metrics(y_true, y_score, k_percentages):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]

    total_pos = np.sum(y_true_sorted)
    n = len(y_true_sorted)

    metrics = {}

    for k_pct in k_percentages:
        k = int(np.ceil(n * (k_pct / 100.0)))
        k = max(1, min(k, n))

        pos_k = np.sum(y_true_sorted[:k])

        metrics[f"precision_at_{k_pct}_pct"] = float(pos_k / k)
        metrics[f"recall_at_{k_pct}_pct"] = float(
            pos_k / total_pos if total_pos > 0 else 0.0
        )
        metrics[f"ndcg_at_{k_pct}_pct"] = float(
            ndcg_score(
                y_true.reshape(1, -1),
                y_score.reshape(1, -1),
                k=k
            )
        )

    return metrics


# =====================================================================================
# Parse training hyperparameters from folder name
# =====================================================================================

def parse_subgraphormer_folder(folder, args):
    """
    Example folder:
    lr-0.01_hid-256_do-0.2_layers-3_pool-mean_metric-prauc_heads-1_focal_g-2.0
    """
    parts = folder.split("_")

    for p in parts:
        try:
            if p.startswith("hid-"):
                args.hidden_channels = int(p.split("-")[1])
            elif p.startswith("layers-"):
                args.num_layers = int(p.split("-")[1])
            elif p.startswith("heads-"):
                args.nhead = int(p.split("-")[1])
            elif p.startswith("do-"):
                args.dropout = float(p.split("-")[1])
            elif p.startswith("pool-"):
                args.pool = p.split("-")[1]
        except Exception:
            pass

    return args


# =====================================================================================
# Main
# =====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset + transform
    # -------------------------
    transform = get_subgraphormer_transform(args)

    if args.dataset_type == "ogb":
        dataset = PygGraphPropPredDataset(
            name=args.dataset,
            root="data/OGB",
            transform=transform
        )
        split_idx = dataset.get_idx_split()
        test_dataset = dataset[split_idx["test"]]
        num_tasks = dataset.num_tasks
        num_features = dataset.num_features
        is_binary = True
    else:
        dataset = TUDataset(
            root="data/TUDataset",
            name=args.dataset,
            transform=transform
        )

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        test_dataset = dataset[indices[: len(dataset) // 10]]

        num_tasks = dataset.num_classes
        num_features = dataset.num_features
        is_binary = num_tasks == 2

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------
    # Results directory
    # -------------------------
    base_path = os.path.join(
        "results_weighted",
        args.dataset,
        "subgraphormer"
    )

    if not os.path.exists(base_path):
        print("No subgraphormer results found.")
        return

    folders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
        and f not in ["summary", "topk_results"]
    ]

    for folder in folders:
        print(f"\nProcessing {folder}")
        exp_path = os.path.join(base_path, folder)
        out_dir = os.path.join(exp_path, "topk_results")
        os.makedirs(out_dir, exist_ok=True)

        args = parse_subgraphormer_folder(folder, args)

        all_metrics = {
            f"{m}_at_{k}_pct": []
            for k in K_PERCENTAGES
            for m in ["precision", "recall", "ndcg"]
        }

        for run in range(1, args.runs + 1):
            model_path = os.path.join(exp_path, f"run_{run}", "model.pt")
            if not os.path.exists(model_path):
                continue

            model = Subgraphormer(
                num_features,
                num_tasks,
                args,
                dataset_type=args.dataset_type
            ).to(device)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            y_true, y_score = [], []

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data)

                    if is_binary:
                        probs = torch.sigmoid(out.view(-1))
                    else:
                        probs = torch.softmax(out, dim=1)[:, 1]

                    y_true.append(data.y.view(-1).cpu())
                    y_score.append(probs.cpu())

            # y_true = torch.cat(y_true).numpy()
            # y_score = torch.cat(y_score).numpy()

            # run_metrics = calculate_ranking_metrics(
            #     y_true, y_score, K_PERCENTAGES
            # )

            y_true = torch.cat(y_true).view(-1).numpy()
            y_score = torch.cat(y_score).view(-1).numpy()
            
            # --- SAFETY FIX: align lengths ---
            n = min(len(y_true), len(y_score))
            y_true = y_true[:n]
            y_score = y_score[:n]
            
            run_metrics = calculate_ranking_metrics(
                y_true, y_score, K_PERCENTAGES
            )

            for k, v in run_metrics.items():
                all_metrics[k].append(v)

            with open(
                os.path.join(out_dir, f"run_{run}_metrics.json"), "w"
            ) as f:
                json.dump(run_metrics, f, indent=4)

        summary = {
            f"{k}_mean": float(np.mean(v))
            for k, v in all_metrics.items()
            if v
        }
        summary.update({
            f"{k}_std": float(np.std(v))
            for k, v in all_metrics.items()
            if v
        })

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        print(f"Saved Top-K results to {out_dir}")


if __name__ == "__main__":
    main()






# #!/usr/bin/env python3
# import argparse
# import os
# import json
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F

# from tqdm import tqdm
# from sklearn.metrics import ndcg_score
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
# from ogb.graphproppred import PygGraphPropPredDataset

# # --- Subgraphormer imports (MATCH TRAINING) ---
# from subgraphormer_model import Subgraphormer
# from subgraphormer_utils import get_subgraphormer_transform
# from pe_layer import compute_laplacian_pe
# from parse_graphclass import parser_add_main_args


# from torch_geometric.data.storage import GlobalStorage
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
# torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# # =====================================================================================
# # K VALUES (EXACT MATCH)
# # =====================================================================================

# K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))

# # =====================================================================================
# # RANKING METRICS
# # =====================================================================================

# def calculate_ranking_metrics(y_true, y_score, k_percentages):
#     y_true = np.asarray(y_true).reshape(-1)
#     y_score = np.asarray(y_score).reshape(-1)

#     order = np.argsort(y_score)[::-1]
#     y_true_sorted = y_true[order]

#     total_pos = np.sum(y_true_sorted)
#     n = len(y_true_sorted)

#     metrics = {}
#     for k_pct in k_percentages:
#         k = int(np.ceil(n * (k_pct / 100.0)))
#         k = max(1, min(k, n))

#         pos_k = np.sum(y_true_sorted[:k])

#         metrics[f"precision_at_{k_pct}_pct"] = float(pos_k / k)
#         metrics[f"recall_at_{k_pct}_pct"] = float(
#             pos_k / total_pos if total_pos > 0 else 0.0
#         )
#         metrics[f"ndcg_at_{k_pct}_pct"] = float(
#             ndcg_score(
#                 y_true.reshape(1, -1),
#                 y_score.reshape(1, -1),
#                 k=k
#             )
#         )

#     return metrics

# # =====================================================================================
# # MAIN
# # =====================================================================================

# def main():
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--dataset", required=True)
#     # parser.add_argument("--device", type=int, default=0)
#     # parser.add_argument("--runs", type=int, default=3)
#     # parser.add_argument("--batch_size", type=int, default=32)

#     # # needed to rebuild Subgraphormer exactly
#     # parser.add_argument("--lap_pe_dim", type=int, default=0)
#     # parser.add_argument("--nhead", type=int, default=4)
#     # parser.add_argument("--hidden_channels", type=int, default=128)
#     # parser.add_argument("--num_layers", type=int, default=6)
#     # parser.add_argument("--pool", type=str, default="mean")
#     # parser.add_argument("--dataset_type", type=str, default="tu")

# #     parser = argparse.ArgumentParser()

# # # -----------------------
# # # Core
# # # -----------------------
# #     parser.add_argument("--dataset", required=True)
# #     parser.add_argument("--dataset_type", type=str, default="tu")
# #     parser.add_argument("--device", type=int, default=0)
# #     parser.add_argument("--runs", type=int, default=3)
# #     parser.add_argument("--batch_size", type=int, default=32)
    
# #     # -----------------------
# #     # Subgraphormer REQUIRED args
# #     # (must match training defaults)
# #     # -----------------------
# #     parser.add_argument("--subgraphormer_aggs", type=str, default="sum,mean,max")
# #     parser.add_argument("--subgraphormer_num_eigen_vectors", type=int, default=8)
# #     parser.add_argument("--subgraphormer_use_sum_pooling", action="store_true")
    
# #     # -----------------------
# #     # Model shape (used only to rebuild model)
# #     # -----------------------
# #     parser.add_argument("--hidden_channels", type=int, default=128)
# #     parser.add_argument("--num_layers", type=int, default=6)
# #     parser.add_argument("--nhead", type=int, default=4)
# #     parser.add_argument("--pool", type=str, default="mean")
    
# #     # -----------------------
# #     # Positional Encoding
# #     # -----------------------
# #     parser.add_argument("--lap_pe_dim", type=int, default=0)
# #     # -----------------------
# # # REQUIRED training flags (safe defaults for eval)
# # # -----------------------
# #     parser.add_argument("--use_residual", action="store_false", default=False)
# #     parser.add_argument("--use_bn", action="store_false", default=False)
# #     parser.add_argument("--dropout", type=float, default=0.0)
# #     parser.add_argument("--weight_decay", type=float, default=0.0)
# #     parser.add_argument("--epochs", type=int, default=1)
# #     parser.add_argument("--metric", type=str, default="rocauc")
# #     parser.add_argument("--seed", type=int, default=42)



# #     args = parser.parse_args()


#     parser = argparse.ArgumentParser()

# # Reuse ALL training-time arguments
#     parser_add_main_args(parser)
    
#     # -----------------------
#     # Eval-only arguments
#     # -----------------------
#     # parser.add_argument("--dataset", required=True)
#     # parser.add_argument("--dataset_type", type=str, default="tu")
#     # parser.add_argument("--device", type=int, default=0)
#     # parser.add_argument("--runs", type=int, default=3)
#     # parser.add_argument("--batch_size", type=int, default=32)
    
#     args = parser.parse_args()


#     device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

#     # ------------------------------------------------------------------
#     # DATASET (MATCH TRAINING)
#     # ------------------------------------------------------------------
#     transform = get_subgraphormer_transform(args)

#     if args.dataset_type == "ogb":
#         dataset = PygGraphPropPredDataset(
#             name=args.dataset,
#             root="data/OGB",
#             transform=transform
#         )

#         if args.lap_pe_dim > 0:
#             for g in tqdm(dataset, desc="Computing LapPE"):
#                 compute_laplacian_pe(g, args.lap_pe_dim)

#         test_idx = dataset.get_idx_split()["test"]
#         test_dataset = dataset[test_idx]
#         num_tasks = dataset.num_tasks
#         is_binary = True

#     else:  # TU DATASETS
#         dataset = TUDataset(
#             root="data/TUDataset",
#             name=args.dataset,
#             transform=transform
#         )

#         # if args.lap_pe_dim > 0:
#         #     for g in tqdm(dataset, desc="Computing LapPE"):
#         #         compute_laplacian_pe(g, args.lap_pe_dim)

#         indices = list(range(len(dataset)))
#         random.shuffle(indices)
#         test_dataset = dataset[indices[: len(dataset) // 10]]

#         num_tasks = dataset.num_classes
#         is_binary = dataset.num_classes == 2

#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#     # ------------------------------------------------------------------
#     # RESULTS PATH
#     # ------------------------------------------------------------------
#     base_path = os.path.join("results_weighted", args.dataset, "subgraphormer")
#     if not os.path.exists(base_path):
#         print(f"No results found at {base_path}")
#         return

#     exp_folders = [
#         f for f in os.listdir(base_path)
#         if os.path.isdir(os.path.join(base_path, f))
#         and f not in ["summary", "topk_results"]
#     ]

#     # ------------------------------------------------------------------
#     # EVAL LOOP
#     # ------------------------------------------------------------------
#     for folder in exp_folders:
#         print(f"\nProcessing {folder}")

#         exp_path = os.path.join(base_path, folder)
#         out_dir = os.path.join(exp_path, "topk_results")
#         os.makedirs(out_dir, exist_ok=True)

#         all_metrics = {
#             f"{m}_at_{k}_pct": []
#             for k in K_PERCENTAGES
#             for m in ["precision", "recall", "ndcg"]
#         }

#         for run in range(1, args.runs + 1):
#             model_path = os.path.join(exp_path, f"run_{run}", "model.pt")
#             if not os.path.exists(model_path):
#                 continue

#             # --- Load model ---
#             model = Subgraphormer(
#                 num_features=dataset.num_features,
#                 num_classes=num_tasks,
#                 args=args,
#                 dataset_type=args.dataset_type
#             ).to(device)

#             model.load_state_dict(torch.load(model_path, map_location=device))
#             model.eval()

#             y_true, y_score = [], []

#             with torch.no_grad():
#                 for data in test_loader:
#                     data = data.to(device)
#                     out = model(data)  # <<< Subgraphormer forward

#                     if is_binary:
#                         probs = torch.sigmoid(out).view(-1)
#                     else:
#                         probs = torch.softmax(out, dim=1)[:, 1]

#                     y_score.append(probs.cpu())
#                     y_true.append(data.y.view(-1).cpu())

#             metrics = calculate_ranking_metrics(
#                 torch.cat(y_true).numpy(),
#                 torch.cat(y_score).numpy(),
#                 K_PERCENTAGES
#             )

#             with open(os.path.join(out_dir, f"run_{run}_metrics.json"), "w") as f:
#                 json.dump(metrics, f, indent=4)

#             for k, v in metrics.items():
#                 all_metrics[k].append(v)

#         # ------------------------------------------------------------------
#         # SUMMARY
#         # ------------------------------------------------------------------
#         summary = {}
#         for k, v in all_metrics.items():
#             if v:
#                 summary[f"{k}_mean"] = float(np.mean(v))
#                 summary[f"{k}_std"] = float(np.std(v))

#         with open(os.path.join(out_dir, "summary.json"), "w") as f:
#             json.dump(summary, f, indent=4)

#         print(f"Saved top-K results to {out_dir}")

# if __name__ == "__main__":
#     main()
