#!/usr/bin/env python
# Fixed main_nodeformer_ordinal.py

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
import sys
import json

from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, class_rand_splits, load_fixed_splits
from eval import evaluate
from logger import Logger
from parse import parser_add_main_args
from nodeformer_adapted import NodeFormerAdapted

# =====================================================================================
#  Section 0: Configuration
# =====================================================================================
ORDINAL_DATASETS = ['amazon-ratings', 'squirrel']

# =====================================================================================
#  Section 1: Helper Functions
# =====================================================================================

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_results(model, embeddings, tsne_embeds, tsne_labels, logits, labels, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings.cpu().numpy())
    np.save(os.path.join(out_dir, 'tsne_embeds.npy'), tsne_embeds)
    np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
    np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
    np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

def plot_tsne(tsne_embeds, tsne_labels, out_dir):
    plt.figure(figsize=(8, 6))
    for c in np.unique(tsne_labels):
        idx = tsne_labels == c
        plt.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=f'Class {c}', alpha=0.6)
    plt.legend()
    plt.title('t-SNE of Node Embeddings (NodeFormer)')
    plt.savefig(os.path.join(out_dir, 'tsne_plot.png'))
    plt.close()

def plot_logits_distribution(logits, labels, out_dir):
    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().numpy().squeeze()
    num_classes = logits.shape[1]
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        if np.any(labels == c):
            plt.hist(logits[labels == c, c], bins=30, alpha=0.5, label=f'Class {c}')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.title('Logits Distribution per Class (NodeFormer)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'logits_distribution.png'))
    plt.close()

# =====================================================================================
#  Section 2: Ordinal Loss Implementations
# =====================================================================================

class DistanceWeightedCrossEntropy(nn.Module):
    def __init__(self, num_classes, alpha=1.0, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        dist = torch.abs(torch.arange(num_classes).float().unsqueeze(1) - 
                         torch.arange(num_classes).float().unsqueeze(0))
        self.register_buffer('weight_mat', 1.0 + self.alpha * dist)

    def forward(self, logits, targets):
        device = logits.device
        if self.weight_mat.device != device:
            self.weight_mat = self.weight_mat.to(device)
        
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        pred_class = logits.argmax(dim=1)
        weights = self.weight_mat[targets, pred_class]
        weighted_loss = ce_loss * weights
        
        return weighted_loss.mean() if self.reduction == 'mean' else weighted_loss.sum()


class EarthMoverDistanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        N, C = logits.shape
        device = logits.device
        
        probs = F.softmax(logits, dim=1)
        cum_probs = torch.cumsum(probs, dim=1)
        
        arange = torch.arange(C, device=device).unsqueeze(0).expand(N, -1)
        targets_expanded = targets.unsqueeze(1)
        cum_target = (arange >= targets_expanded).float()
        
        diff = torch.abs(cum_probs - cum_target)
        per_sample_loss = diff.sum(dim=1)
        
        return per_sample_loss.mean() if self.reduction == 'mean' else per_sample_loss.sum()


class CORALLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        N, C = logits.shape
        device = logits.device
        
        probs = F.softmax(logits, dim=1)
        
        cum_probs = torch.zeros(N, C-1, device=device)
        for k in range(C-1):
            cum_probs[:, k] = probs[:, k+1:].sum(dim=1)
        
        target_cum = torch.zeros(N, C-1, device=device)
        for k in range(C-1):
            target_cum[:, k] = (targets > k).float()
        
        cum_probs = torch.clamp(cum_probs, min=1e-7, max=1-1e-7)
        
        bce_loss = -(target_cum * torch.log(cum_probs) + 
                       (1 - target_cum) * torch.log(1 - cum_probs))
        
        per_sample_loss = bce_loss.sum(dim=1)
        
        return per_sample_loss.mean() if self.reduction == 'mean' else per_sample_loss.sum()

# =====================================================================================
#  Section 3: Ordinal Prediction and Metrics
# =====================================================================================

def get_ordinal_predictions(logits, method='expected_value'):
    probs = F.softmax(logits, dim=1)
    
    if method == 'argmax':
        return probs.argmax(dim=1)
    elif method == 'expected_value':
        C = logits.shape[1]
        class_indices = torch.arange(C, device=logits.device).float()
        pred = (probs * class_indices).sum(dim=1)
        return pred.round().long()
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_ordinal_metrics(y_true_tensor, y_pred_tensor):
    y_true = y_true_tensor.cpu().numpy() if torch.is_tensor(y_true_tensor) else y_true_tensor
    y_pred = y_pred_tensor.cpu().numpy() if torch.is_tensor(y_pred_tensor) else y_pred_tensor
    
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    try:
        spearman_corr = spearmanr(y_true, y_pred).correlation
        if spearman_corr is None or np.isnan(spearman_corr):
            spearman_corr = 0.0
    except Exception:
        spearman_corr = 0.0
    
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        qwk = 0.0
    
    acc_exact = np.mean(y_true == y_pred)
    acc1 = np.mean(np.abs(y_true - y_pred) <= 1)
    acc2 = np.mean(np.abs(y_true - y_pred) <= 2)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman": float(spearman_corr),
        "qwk": float(qwk),
        "acc_exact": float(acc_exact),
        "acc1": float(acc1),
        "acc2": float(acc2)
    }

# =====================================================================================
#  Section 4: Custom Evaluate Function for Ordinal
# =====================================================================================

def evaluate_ordinal(model, dataset, split_idx, criterion, args):
    model.eval()
    
    with torch.no_grad():
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        predictions = get_ordinal_predictions(out, method='expected_value')
        y_true = dataset.label.squeeze(1)
        
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        
        if args.metric == 'acc':
            train_metric = (predictions[train_idx] == y_true[train_idx]).float().mean().item()
            val_metric = (predictions[valid_idx] == y_true[valid_idx]).float().mean().item()
            test_metric = (predictions[test_idx] == y_true[test_idx]).float().mean().item()
        
        elif args.metric == 'mae':
            train_mae = torch.abs(predictions[train_idx].float() - y_true[train_idx].float()).mean().item()
            val_mae = torch.abs(predictions[valid_idx].float() - y_true[valid_idx].float()).mean().item()
            test_mae = torch.abs(predictions[test_idx].float() - y_true[test_idx].float()).mean().item()
            train_metric, val_metric, test_metric = -train_mae, -val_mae, -test_mae
        
        elif args.metric == 'rmse':
            train_rmse = torch.sqrt(((predictions[train_idx].float() - y_true[train_idx].float()) ** 2).mean()).item()
            val_rmse = torch.sqrt(((predictions[valid_idx].float() - y_true[valid_idx].float()) ** 2).mean()).item()
            test_rmse = torch.sqrt(((predictions[test_idx].float() - y_true[test_idx].float()) ** 2).mean()).item()
            train_metric, val_metric, test_metric = -train_rmse, -val_rmse, -test_rmse
        
        elif args.metric == 'qwk':
            train_qwk = cohen_kappa_score(y_true[train_idx].cpu().numpy(), predictions[train_idx].cpu().numpy(), weights="quadratic")
            val_qwk = cohen_kappa_score(y_true[valid_idx].cpu().numpy(), predictions[valid_idx].cpu().numpy(), weights="quadratic")
            test_qwk = cohen_kappa_score(y_true[test_idx].cpu().numpy(), predictions[test_idx].cpu().numpy(), weights="quadratic")
            train_metric, val_metric, test_metric = train_qwk, val_qwk, test_qwk
        
        elif args.metric == 'spearman':
            train_spearman = spearmanr(y_true[train_idx].cpu().numpy(), predictions[train_idx].cpu().numpy()).correlation
            val_spearman = spearmanr(y_true[valid_idx].cpu().numpy(), predictions[valid_idx].cpu().numpy()).correlation
            test_spearman = spearmanr(y_true[test_idx].cpu().numpy(), predictions[test_idx].cpu().numpy()).correlation
            train_spearman = 0.0 if train_spearman is None or np.isnan(train_spearman) else train_spearman
            val_spearman = 0.0 if val_spearman is None or np.isnan(val_spearman) else val_spearman
            test_spearman = 0.0 if test_spearman is None or np.isnan(test_spearman) else test_spearman
            train_metric, val_metric, test_metric = train_spearman, val_spearman, test_spearman
        
        else:
            train_metric = (predictions[train_idx] == y_true[train_idx]).float().mean().item()
            val_metric = (predictions[valid_idx] == y_true[valid_idx]).float().mean().item()
            test_metric = (predictions[test_idx] == y_true[test_idx]).float().mean().item()
            
        # train_loss = criterion(out[train_idx], y_true[train_idx]).item()
        val_loss = criterion(out[valid_idx], y_true[valid_idx]).item()
    
    # return train_metric, val_metric, test_metric, train_loss
    return train_metric, val_metric, test_metric, val_loss

# =====================================================================================
#  Section 5: Main Training and Evaluation Function
# =====================================================================================

def train_and_evaluate(args):
    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    args.device = device
    print(f"Using device: {device}")

    args.gnn = 'nodeformer'

    # Load dataset
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if args.rand_split:
        split_idx_lst = [
            dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
            for _ in range(args.runs)
        ]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f'Dataset: {args.dataset}, Nodes: {n}, Features: {d}, Classes: {c}')

    # Check if ordinal
    is_ordinal = args.dataset in ORDINAL_DATASETS
    if is_ordinal:
        print(f">>> Dataset '{args.dataset}' identified as ORDINAL <<<")

    # Prepare graph
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    # Model
    model = NodeFormerAdapted(
        in_channels=d,
        hidden_channels=args.hidden_channels,
        out_channels=c,
        local_layers=args.local_layers,
        dropout=args.dropout,
        heads=args.num_heads,
        res=args.res,
        ln=args.ln,
        jk=args.jk
    ).to(device)

    print(f"Model: NodeFormer, Parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss
    if is_ordinal:
        if args.ordinal_loss == 'dwce':
            criterion = DistanceWeightedCrossEntropy(c, alpha=args.dwce_alpha)
        elif args.ordinal_loss == 'emd':
            criterion = EarthMoverDistanceLoss()
        elif args.ordinal_loss == 'coral':
            criterion = CORALLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    eval_func = {'prauc': eval_rocauc, 'rocauc': eval_rocauc}.get(args.metric, eval_acc)

    # 🔹 NEW: store all runs
    all_runs = []

    exp_name = f"l_{args.local_layers}-h_{args.hidden_channels}-d_{args.dropout}-heads_{args.num_heads}"
    if is_ordinal:
        exp_name += f"-ord_{args.ordinal_loss}"
        if args.ordinal_loss == 'dwce':
            exp_name += f"-a_{args.dwce_alpha}"

    for run in range(args.runs):
        print(f"\n{'='*80}")
        print(f"Run {run+1}/{args.runs}")
        print(f"{'='*80}")

        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        valid_idx = split_idx['valid'].to(device)
        test_idx = split_idx['test'].to(device)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val = float('-inf')
        best_test = float('-inf')
        best_epoch = 0
        best_model_state = None
        patience_counter = 0

        # 🔹 NEW: per-epoch history
        epoch_history = {
            "train": [],
            "val": [],
            "test": [],
            "val_loss": []
        }

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            if is_ordinal:
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            else:
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if is_ordinal:
                    result = evaluate_ordinal(model, dataset, split_idx, criterion, args)
                else:
                    result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

            train_metric, val_metric, test_metric, val_loss = result

            # 🔹 NEW: save metrics per epoch
            epoch_history["train"].append(float(train_metric))
            epoch_history["val"].append(float(val_metric))
            epoch_history["test"].append(float(test_metric))
            epoch_history["val_loss"].append(float(val_loss))

            if val_metric > best_val:
                best_val = val_metric
                best_test = test_metric
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % args.display_step == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train {train_metric:.4f} | "
                    f"Val {val_metric:.4f} | "
                    f"Test {test_metric:.4f}"
                )

            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\nBest Epoch: {best_epoch}, Best Val: {best_val:.4f}, Best Test: {best_test:.4f}")

        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()

        with torch.no_grad():
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

        if is_ordinal:
            y_pred = get_ordinal_predictions(out, method='expected_value')
        else:
            y_pred = out.argmax(dim=-1)

        accuracy = (y_pred[test_idx] == dataset.label.squeeze(1)[test_idx]).float().mean().item()
        probs = F.softmax(out[test_idx], dim=1)

        try:
            roc_auc = roc_auc_score(
                dataset.label.squeeze(1)[test_idx].cpu().numpy(),
                probs.cpu().numpy(),
                multi_class='ovr'
            )
        except Exception:
            roc_auc = 0.0

        # 🔹 NEW: unified per-run metrics
        run_results = {
            "run": run,
            "best_epoch": best_epoch,
            "best_val": float(best_val),
            "best_test": float(best_test),
            "test_accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "epoch_history": epoch_history
        }

        if is_ordinal:
            ordinal_metrics = compute_ordinal_metrics(
                dataset.label.squeeze(1)[test_idx].cpu().numpy(),
                y_pred[test_idx].cpu().numpy()
            )
            run_results.update({f"ordinal_{k}": v for k, v in ordinal_metrics.items()})

        # 🔹 SAME directory as before
        if args.output_dir is not None:
            out_dir = os.path.join(args.output_dir, f'run_{run+1}')
        else:
            out_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/run_{run+1}'

        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(run_results, f, indent=2)

        all_runs.append(run_results)

    # ===== SUMMARY =====
    summary = {}
    numeric_keys = [k for k, v in all_runs[0].items() if isinstance(v, (int, float))]
    for k in numeric_keys:
        values = [r[k] for r in all_runs]
        summary[f"{k}_mean"] = float(np.mean(values))
        summary[f"{k}_std"] = float(np.std(values))

    summary_dir = (
        args.output_dir
        if args.output_dir is not None
        else f'results/{args.dataset}/{args.gnn}/{exp_name}'
    )
    os.makedirs(summary_dir, exist_ok=True)

    with open(os.path.join(summary_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(summary_dir, "all_runs.json"), "w") as f:
        json.dump(all_runs, f, indent=2)

    print(f"\nResults saved to: {summary_dir}")




# def train_and_evaluate(args):
#     fix_seed(args.seed)
#     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
#     args.device = device
#     print(f"Using device: {device}")

#     args.gnn = 'nodeformer'

#     # Load dataset
#     dataset = load_dataset(args.data_dir, args.dataset)
#     if len(dataset.label.shape) == 1:
#         dataset.label = dataset.label.unsqueeze(1)

#     if args.rand_split:
#         split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
#                          for _ in range(args.runs)]
#     else:
#         split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

#     dataset.label = dataset.label.to(device)

#     n = dataset.graph['num_nodes']
#     c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
#     d = dataset.graph['node_feat'].shape[1]

#     print(f'Dataset: {args.dataset}, Nodes: {n}, Features: {d}, Classes: {c}')
    
#     # Check if ordinal
#     is_ordinal = args.dataset in ORDINAL_DATASETS
#     if is_ordinal:
#         print(f">>> Dataset '{args.dataset}' identified as ORDINAL <<<")
#         unique_labels = dataset.label.unique().cpu().numpy()
#         print(f"Unique labels: {unique_labels}")
#         if not np.array_equal(np.sort(unique_labels), np.arange(len(unique_labels))):
#             print("WARNING: Labels are not 0-indexed consecutive integers!")

#     # Prepare graph
#     dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
#     dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
#     dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

#     # Initialize model
#     print("Instantiating NodeFormerAdapted model...")
#     model = NodeFormerAdapted(
#         in_channels=d,
#         hidden_channels=args.hidden_channels,
#         out_channels=c,
#         local_layers=args.local_layers,
#         dropout=args.dropout,
#         heads=args.num_heads,
#         res=args.res,
#         ln=args.ln,
#         jk=args.jk
#     ).to(device)
    
#     print(f"Model: NodeFormer, Parameters: {sum(p.numel() for p in model.parameters())}")

#     # Loss function selection
#     if is_ordinal:
#         if args.ordinal_loss == 'dwce':
#             criterion = DistanceWeightedCrossEntropy(num_classes=c, alpha=args.dwce_alpha)
#             print(f"Using DistanceWeightedCrossEntropy (alpha={args.dwce_alpha})")
#         elif args.ordinal_loss == 'emd':
#             criterion = EarthMoverDistanceLoss()
#             print(f"Using EarthMoverDistanceLoss")
#         elif args.ordinal_loss == 'coral':
#             criterion = CORALLoss()
#             print(f"Using CORALLoss")
#         else:
#             criterion = nn.CrossEntropyLoss()
#             print(f"Using standard CrossEntropyLoss")
#     else:
#         if args.dataset in ('questions',):
#             criterion = nn.BCEWithLogitsLoss()
#         else:
#             criterion = nn.CrossEntropyLoss()  # ✅ Use CrossEntropyLoss instead of NLLLoss
            
#     eval_func = {'prauc': eval_rocauc, 'rocauc': eval_rocauc}.get(args.metric, eval_acc)
#     logger = Logger(args.runs, args)
    
#     all_acc, all_roc_auc, all_pr_auc, all_reports = [], [], [], []
#     ordinal_metric_keys = ['mae', 'rmse', 'spearman', 'qwk', 'acc_exact', 'acc1', 'acc2']
#     metric_lists = {k: [] for k in ordinal_metric_keys}

#     exp_name = f"l_{args.local_layers}-h_{args.hidden_channels}-d_{args.dropout}-heads_{args.num_heads}"
#     if is_ordinal:
#         exp_name += f"-ord_{args.ordinal_loss}"
#         if args.ordinal_loss == 'dwce':
#             exp_name += f"-a_{args.dwce_alpha}"

#     # Training loop
#     for run in range(args.runs):
#         print(f"\n{'='*80}")
#         print(f"Run {run+1}/{args.runs}")
#         print(f"{'='*80}")
        
#         split_idx = split_idx_lst[0] if args.dataset in ('cora', 'citeseer', 'pubmed') else split_idx_lst[run]
#         train_idx = split_idx['train'].to(device)
#         valid_idx = split_idx['valid'].to(device)
#         test_idx = split_idx['test'].to(device)
        
#         print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
        
#         model.reset_parameters()
#         optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
#         best_val = float('-inf')
#         best_test = float('-inf')
#         best_epoch = 0
#         best_model_state = None
#         patience_counter = 0

#         for epoch in range(args.epochs):
#             model.train()
#             optimizer.zero_grad()
            
#             # ✅ Get fresh logits every time
#             out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

#             # ✅ Compute loss with raw logits
#             if is_ordinal:
#                 targets_train = dataset.label.squeeze(1)[train_idx].long()
#                 loss = criterion(out[train_idx], targets_train)
#             elif args.dataset in ('questions',):
#                 if dataset.label.shape[1] == 1:
#                     true_label = F.one_hot(dataset.label, num_classes=c).squeeze(1)
#                 else:
#                     true_label = dataset.label
#                 loss = criterion(out[train_idx], true_label[train_idx].to(torch.float))
#             else:
#                 # ✅ CrossEntropyLoss takes raw logits, not log_softmax
#                 loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            
#             loss.backward()
#             optimizer.step()

#             # Evaluation
#             with torch.no_grad():
#                 if is_ordinal:
#                     result = evaluate_ordinal(model, dataset, split_idx, criterion, args)
#                 else:
#                     result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

#             val_loss = result[3]  # Extract validation loss
#             logger.add_result(run, result)

#             train_metric, val_metric, test_metric, train_loss = result

#             if val_metric > best_val:
#                 best_val = val_metric
#                 best_test = test_metric
#                 best_epoch = epoch
#                 best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#                 patience_counter = 0
#             else:
#                 patience_counter += 1

#             if epoch % args.display_step == 0:
#                 # Smart display based on metric type
#                 if args.metric in ['mae', 'rmse']:
#                     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#                           f'Train: {abs(train_metric):.4f}, '
#                           f'Val: {abs(val_metric):.4f}, '
#                           f'Test: {abs(test_metric):.4f}')
#                 elif args.metric in ['qwk', 'spearman']:
#                     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#                           f'Train: {train_metric:.4f}, '
#                           f'Val: {val_metric:.4f}, '
#                           f'Test: {test_metric:.4f}')
#                 else:
#                     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#                           f'Train: {100*train_metric:.2f}%, '
#                           f'Val: {100*val_metric:.2f}%, '
#                           f'Test: {100*test_metric:.2f}%')
                    
#             scheduler.step(val_loss)
            
#             if args.early_stopping > 0 and patience_counter >= args.early_stopping:
#                 print(f"Early stopping at epoch {epoch}")
#                 break
        
#         # Display best results
#         if args.metric in ['mae', 'rmse']:
#             print(f'\nBest Epoch: {best_epoch}, Best Val: {abs(best_val):.4f}, Best Test: {abs(best_test):.4f}')
#         elif args.metric in ['qwk', 'spearman']:
#             print(f'\nBest Epoch: {best_epoch}, Best Val: {best_val:.4f}, Best Test: {best_test:.4f}')
#         else:
#             print(f'\nBest Epoch: {best_epoch}, Best Val: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')
            
#         model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
#         # Final evaluation
#         model.eval()
#         with torch.no_grad():
#             # ✅ Get fresh logits for final eval
#             out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

#         ordinal_metrics = {}
        
#         if is_ordinal:
#             y_true_np = dataset.label[test_idx].cpu().numpy().squeeze()
#             y_pred_tensor = get_ordinal_predictions(out[test_idx], method='expected_value')
#             y_pred_np = y_pred_tensor.cpu().numpy()
            
#             ordinal_metrics = compute_ordinal_metrics(y_true_np, y_pred_np)
#             accuracy = ordinal_metrics['acc_exact']
#             report = classification_report(y_true_np, y_pred_np, output_dict=True, zero_division=0)
            
#             probs = F.softmax(out[test_idx], dim=1)
#             try:
#                 roc_auc = roc_auc_score(y_true_np, probs.cpu().numpy(), multi_class='ovr')
#                 y_true_binarized = label_binarize(y_true_np, classes=range(c))
#                 pr_auc = average_precision_score(y_true_binarized, probs.cpu().numpy())
#             except Exception:
#                 roc_auc = 0.0
#                 pr_auc = 0.0

#             print(f'\nFinal Test Accuracy (Exact): {100*accuracy:.2f}%')
#             print(f'AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}')
#             print("--- Ordinal Metrics ---")
#             print(f"MAE: {ordinal_metrics['mae']:.4f}")
#             print(f"RMSE: {ordinal_metrics['rmse']:.4f}")
#             print(f"Spearman: {ordinal_metrics['spearman']:.4f}")
#             print(f"QWK: {ordinal_metrics['qwk']:.4f}")
#             print(f"Acc@1: {100*ordinal_metrics['acc1']:.2f}%")

#         else:
#             probs = F.softmax(out[test_idx], dim=1)
#             y_pred = probs.argmax(dim=-1)
#             y_true_test = dataset.label[test_idx]
            
#             accuracy = (y_pred == y_true_test.squeeze()).float().mean().item()
#             report = classification_report(y_true_test.cpu().numpy(), y_pred.cpu().numpy(), output_dict=True, zero_division=0)
            
#             if c > 2:
#                 roc_auc = roc_auc_score(y_true_test.cpu().numpy(), probs.cpu().numpy(), multi_class='ovr')
#                 y_true_binarized = label_binarize(y_true_test.cpu().numpy(), classes=range(c))
#                 pr_auc = average_precision_score(y_true_binarized, probs.cpu().numpy())
#             else:
#                 roc_auc = roc_auc_score(y_true_test.cpu().numpy(), probs[:,1].cpu().numpy())
#                 pr_auc = average_precision_score(y_true_test.cpu().numpy(), probs[:,1].cpu().numpy())

#             print(f'Final Test Accuracy: {100*accuracy:.2f}%, AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}')
            
#         all_acc.append(accuracy)
#         all_roc_auc.append(roc_auc)
#         all_pr_auc.append(pr_auc)
#         all_reports.append(report)
        
#         if ordinal_metrics:
#             for k, v in ordinal_metrics.items():
#                 metric_lists[k].append(v)

#         # Save results
#         with torch.no_grad():
#             embeddings = model.get_embeddings(dataset.graph['node_feat'], dataset.graph['edge_index'])
#             embeddings_cpu = embeddings[test_idx].cpu().numpy()
#             labels_cpu = dataset.label[test_idx].cpu().numpy().squeeze()
#             # tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(labels_cpu)-1))
#             # tsne_embeds = tsne.fit_transform(embeddings_cpu)
#             tsne_embeds=None

#         # out_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/run_{run+1}'
#         # Use custom output directory if provided
#         if args.output_dir is not None:
#             base_out_dir = os.path.join(args.output_dir, f'run_{run+1}')
#         else:
#             base_out_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/run_{run+1}'
#         out_dir = base_out_dir
#         os.makedirs(out_dir, exist_ok=True)

#         save_results(model, embeddings[test_idx], tsne_embeds, labels_cpu, 
#                     out[test_idx], dataset.label[test_idx], args, out_dir)
#         # plot_tsne(tsne_embeds, labels_cpu, out_dir)
#         # plot_logits_distribution(out[test_idx], dataset.label[test_idx], out_dir)

#         metrics = {'accuracy': accuracy, 'auc_roc': roc_auc, 'auc_pr': pr_auc, 'classification_report': report}
#         if ordinal_metrics:
#             metrics.update({f'ordinal_{k}': v for k, v in ordinal_metrics.items()})
            
#         with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
#             json.dump(metrics, f, indent=2)

#     # Summary over all runs
#     summary = {
#         'accuracy_mean': float(np.mean(all_acc)), 
#         'accuracy_std': float(np.std(all_acc)),
#         'auc_roc_mean': float(np.mean(all_roc_auc)), 
#         'auc_roc_std': float(np.std(all_roc_auc)),
#         'auc_pr_mean': float(np.mean(all_pr_auc)), 
#         'auc_pr_std': float(np.std(all_pr_auc)),
#     }
    
#     if is_ordinal:
#         print("\n" + "="*80)
#         print("AGGREGATED ORDINAL METRICS (mean ± std)")
#         print("="*80)
#         for k in ordinal_metric_keys:
#             values = metric_lists[k]
#             if values:
#                 mean_val = float(np.mean(values))
#                 std_val = float(np.std(values))
#                 summary[f'ordinal_{k}_mean'] = mean_val
#                 summary[f'ordinal_{k}_std'] = std_val
                
#                 if k in ['mae', 'rmse']:
#                     print(f"ordinal_{k}: {mean_val:.4f} ± {std_val:.4f}")
#                 elif k in ['spearman', 'qwk']:
#                     print(f"ordinal_{k}: {mean_val:.4f} ± {std_val:.4f}")
#                 else:
#                     print(f"ordinal_{k}: {100*mean_val:.2f} ± {100*std_val:.2f}%")
    
#     # summary_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/'
#     summary_dir = args.output_dir if args.output_dir is not None else f'results/{args.dataset}/{args.gnn}/{exp_name}/'

#     os.makedirs(summary_dir, exist_ok=True)
#     with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     print("\n" + "="*80)
#     print("FINAL SUMMARY")
#     print("="*80)
#     print(f"Accuracy: {100*summary['accuracy_mean']:.2f} ± {100*summary['accuracy_std']:.2f}%")
#     print(f"ROC-AUC: {summary['auc_roc_mean']:.4f} ± {summary['auc_roc_std']:.4f}")
#     print(f"PR-AUC: {summary['auc_pr_mean']:.4f} ± {summary['auc_pr_std']:.4f}")
#     print(f"\nResults saved to: {summary_dir}")


# =====================================================================================
#  Section 6: Script Entry Point
# =====================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NodeFormer Training Pipeline with Ordinal Support')
    parser_add_main_args(parser)
    
    # Ordinal-specific arguments
    parser.add_argument('--ordinal_loss', type=str, default='coral', 
                        choices=['coral', 'dwce', 'emd', 'ce'],
                        help="Ordinal loss: coral/dwce/emd/ce")
    parser.add_argument('--dwce_alpha', type=float, default=0.5,
                        help="Alpha for DistanceWeightedCrossEntropy")
    parser.add_argument('--early_stopping', type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Optional path to save results (overrides default results/<dataset>/<model>/...)")

    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("NODEFORMER TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: NodeFormer")
    print(f"Validation Metric: {args.metric.upper()}")
    if args.dataset in ORDINAL_DATASETS:
        print(f"Ordinal Loss: {args.ordinal_loss}")
        if args.ordinal_loss == 'dwce':
            print(f"  - DWCE Alpha: {args.dwce_alpha}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Hidden Channels: {args.hidden_channels}")
    print(f"Local Layers: {args.local_layers}")
    print(f"Attention Heads: {args.num_heads}")
    print(f"Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Runs: {args.runs}")
    print(f"Seed: {args.seed}")
    if args.early_stopping > 0:
        print(f"Early Stopping: {args.early_stopping} epochs")
    print("="*80 + "\n")
    
    train_and_evaluate(args)



# #!/usr/bin/env python
# # Save this file as main_nodeformer_ordinal.py
# # This is a dedicated training script for the NodeFormer model,
# # now with integrated ordinal regression functionalities.

# import argparse
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
# from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report, balanced_accuracy_score
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import os
# import sys
# import json
# from sklearn.preprocessing import label_binarize

# # --- ORDINAL INTEGRATION ---
# from sklearn.metrics import cohen_kappa_score
# from scipy.stats import spearmanr
# # -------------------------

# # Import necessary utilities from your project structure
# from dataset import load_dataset
# from data_utils import eval_acc, eval_rocauc, class_rand_splits, load_fixed_splits
# from eval import evaluate
# from logger import Logger
# from parse import parser_add_main_args

# # Import our adapted NodeFormer model
# from nodeformer_adapted import NodeFormerAdapted

# # =====================================================================================
# #  Section 0: Configuration
# # =====================================================================================

# ### --- ORDINAL INTEGRATION --- ###
# ORDINAL_DATASETS = ['amazon-ratings', 'squirrel']
# ### ----------------------------- ###

# # =====================================================================================
# #  Section 1: Helper Functions
# # =====================================================================================

# def fix_seed(seed=42):
#     """Sets the seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
# def save_results(model, embeddings, tsne_embeds, tsne_labels, logits, labels, args, out_dir):
#     """Saves all artifacts from a training run."""
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
#     np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings.cpu().numpy())
#     np.save(os.path.join(out_dir, 'tsne_embeds.npy'), tsne_embeds)
#     np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
#     np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
#     np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

# def plot_tsne(tsne_embeds, tsne_labels, out_dir):
#     """Generates and saves a t-SNE plot of node embeddings."""
#     plt.figure(figsize=(8, 6))
#     for c in np.unique(tsne_labels):
#         idx = tsne_labels == c
#         plt.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=f'Class {c}', alpha=0.6)
#     plt.legend()
#     plt.title('t-SNE of Node Embeddings (NodeFormer)')
#     plt.savefig(os.path.join(out_dir, 'tsne_plot.png'))
#     plt.close()

# def plot_logits_distribution(logits, labels, out_dir):
#     """Generates and saves a histogram of the model's output logits."""
#     logits = logits.cpu().detach().numpy()
#     labels = labels.cpu().numpy().squeeze()
#     num_classes = logits.shape[1]
#     plt.figure(figsize=(8, 6))
#     for c in range(num_classes):
#         if np.any(labels == c):
#             plt.hist(logits[labels == c, c], bins=30, alpha=0.5, label=f'Class {c}')
#     plt.xlabel('Logit Value')
#     plt.ylabel('Frequency')
#     plt.title('Logits Distribution per Class (NodeFormer)')
#     plt.legend()
#     plt.savefig(os.path.join(out_dir, 'logits_distribution.png'))
#     plt.close()

# # =====================================================================================
# #  Section 2: CORRECTED Ordinal Loss Implementations
# # =====================================================================================

# ### --- ORDINAL INTEGRATION --- ###
# class DistanceWeightedCrossEntropy(nn.Module):
#     """
#     Distance-weighted cross-entropy.
#     Penalizes misclassifications proportional to their distance from true class.
#     """
#     def __init__(self, num_classes, alpha=1.0, reduction='mean'):
#         super().__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.reduction = reduction
#         # Distance matrix: dist[i,j] = |i - j|
#         dist = torch.abs(torch.arange(num_classes).float().unsqueeze(1) - 
#                          torch.arange(num_classes).float().unsqueeze(0))
#         # Weight increases with distance: w[i,j] = 1 + alpha * |i - j|
#         self.register_buffer('weight_mat', 1.0 + self.alpha * dist)

#     def forward(self, logits, targets):
#         """
#         logits: [N, C] raw logits
#         targets: [N] class indices
#         """
#         device = logits.device
#         if self.weight_mat.device != device:
#             self.weight_mat = self.weight_mat.to(device)
        
#         log_probs = F.log_softmax(logits, dim=1)
#         ce_loss = F.nll_loss(log_probs, targets, reduction='none') # [N]
#         pred_class = logits.argmax(dim=1) # [N]
#         weights = self.weight_mat[targets, pred_class] # [N]
#         weighted_loss = ce_loss * weights
        
#         if self.reduction == 'mean':
#             return weighted_loss.mean()
#         elif self.reduction == 'sum':
#             return weighted_loss.sum()
#         else:
#             return weighted_loss


# class EarthMoverDistanceLoss(nn.Module):
#     """
#     EMD-like loss using cumulative distribution difference.
#     """
#     def __init__(self, reduction='mean'):
#         super().__init__()
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         """
#         logits: [N, C] raw logits
#         targets: [N] class indices
#         """
#         N, C = logits.shape
#         device = logits.device
        
#         probs = F.softmax(logits, dim=1) # [N, C]
#         cum_probs = torch.cumsum(probs, dim=1) # [N, C]
        
#         arange = torch.arange(C, device=device).unsqueeze(0).expand(N, -1) # [N, C]
#         targets_expanded = targets.unsqueeze(1) # [N, 1]
#         cum_target = (arange >= targets_expanded).float() # [N, C]
        
#         diff = torch.abs(cum_probs - cum_target)
#         per_sample_loss = diff.sum(dim=1) # Sum over classes
        
#         if self.reduction == 'mean':
#             return per_sample_loss.mean()
#         elif self.reduction == 'sum':
#             return per_sample_loss.sum()
#         else:
#             return per_sample_loss


# class CORALLoss(nn.Module):
#     """
#     Simplified CORAL (Consistent Rank Logits) loss.
#     """
#     def __init__(self, reduction='mean'):
#         super().__init__()
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         """
#         logits: [N, C] raw logits
#         targets: [N] class indices in range [0, C-1]
#         """
#         N, C = logits.shape
#         device = logits.device
        
#         probs = F.softmax(logits, dim=1) # [N, C]
        
#         cum_probs = torch.zeros(N, C-1, device=device)
#         for k in range(C-1):
#             cum_probs[:, k] = probs[:, k+1:].sum(dim=1)
        
#         target_cum = torch.zeros(N, C-1, device=device)
#         for k in range(C-1):
#             target_cum[:, k] = (targets > k).float()
        
#         cum_probs = torch.clamp(cum_probs, min=1e-7, max=1-1e-7)
        
#         bce_loss = -(target_cum * torch.log(cum_probs) + 
#                        (1 - target_cum) * torch.log(1 - cum_probs))
        
#         per_sample_loss = bce_loss.sum(dim=1) # Sum over thresholds
        
#         if self.reduction == 'mean':
#             return per_sample_loss.mean()
#         elif self.reduction == 'sum':
#             return per_sample_loss.sum()
#         else:
#             return per_sample_loss
# ### ----------------------------- ###


# # =====================================================================================
# #  Section 3: Ordinal Prediction and Metrics
# # =====================================================================================

# ### --- ORDINAL INTEGRATION --- ###
# def get_ordinal_predictions(logits, method='expected_value'):
#     """
#     Convert logits to ordinal predictions.
#     """
#     probs = F.softmax(logits, dim=1) # [N, C]
    
#     if method == 'argmax':
#         return probs.argmax(dim=1)
#     elif method == 'expected_value':
#         C = logits.shape[1]
#         class_indices = torch.arange(C, device=logits.device).float()
#         pred = (probs * class_indices).sum(dim=1) # [N]
#         return pred.round().long()
#     else:
#         raise ValueError(f"Unknown method: {method}")


# def compute_ordinal_metrics(y_true_tensor, y_pred_tensor):
#     """Compute comprehensive ordinal metrics."""
#     y_true = y_true_tensor.cpu().numpy() if torch.is_tensor(y_true_tensor) else y_true_tensor
#     y_pred = y_pred_tensor.cpu().numpy() if torch.is_tensor(y_pred_tensor) else y_pred_tensor
    
#     y_true = y_true.astype(int).flatten()
#     y_pred = y_pred.astype(int).flatten()

#     mae = np.mean(np.abs(y_true - y_pred))
#     rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
#     try:
#         spearman_corr = spearmanr(y_true, y_pred).correlation
#         if spearman_corr is None or np.isnan(spearman_corr):
#             spearman_corr = 0.0
#     except Exception:
#         spearman_corr = 0.0
    
#     try:
#         qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
#     except Exception:
#         qwk = 0.0
    
#     acc_exact = np.mean(y_true == y_pred)
#     acc1 = np.mean(np.abs(y_true - y_pred) <= 1)
#     acc2 = np.mean(np.abs(y_true - y_pred) <= 2)
    
#     return {
#         "mae": float(mae),
#         "rmse": float(rmse),
#         "spearman": float(spearman_corr),
#         "qwk": float(qwk),
#         "acc_exact": float(acc_exact),
#         "acc1": float(acc1),
#         "acc2": float(acc2)
#     }
# ### ----------------------------- ###

# # =====================================================================================
# #  Section 4: Custom Evaluate Function for Ordinal
# # =====================================================================================

# ### --- ORDINAL INTEGRATION --- ###
# def evaluate_ordinal(model, dataset, split_idx, criterion, args):
#     """
#     Evaluate function specifically for ordinal datasets with NodeFormer.
#     Returns metric based on args.metric (acc, mae, rmse, qwk, spearman)
#     Returns: (train_metric, val_metric, test_metric, train_loss)
#     """
#     model.eval()
    
#     with torch.no_grad():
#         # Forward pass (specific to NodeFormer)
#         out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
            
#         # Get predictions using expected value for ordinal
#         predictions = get_ordinal_predictions(out, method='expected_value')
        
#         # Get true labels
#         y_true = dataset.label.squeeze(1)
        
#         # Compute metric for each split based on args.metric
#         train_idx = split_idx['train']
#         valid_idx = split_idx['valid']
#         test_idx = split_idx['test']
        
#         if args.metric == 'acc':
#             train_metric = (predictions[train_idx] == y_true[train_idx]).float().mean().item()
#             val_metric = (predictions[valid_idx] == y_true[valid_idx]).float().mean().item()
#             test_metric = (predictions[test_idx] == y_true[test_idx]).float().mean().item()
        
#         elif args.metric == 'mae':
#             train_mae = torch.abs(predictions[train_idx].float() - y_true[train_idx].float()).mean().item()
#             val_mae = torch.abs(predictions[valid_idx].float() - y_true[valid_idx].float()).mean().item()
#             test_mae = torch.abs(predictions[test_idx].float() - y_true[test_idx].float()).mean().item()
#             train_metric, val_metric, test_metric = -train_mae, -val_mae, -test_mae
        
#         elif args.metric == 'rmse':
#             train_rmse = torch.sqrt(((predictions[train_idx].float() - y_true[train_idx].float()) ** 2).mean()).item()
#             val_rmse = torch.sqrt(((predictions[valid_idx].float() - y_true[valid_idx].float()) ** 2).mean()).item()
#             test_rmse = torch.sqrt(((predictions[test_idx].float() - y_true[test_idx].float()) ** 2).mean()).item()
#             train_metric, val_metric, test_metric = -train_rmse, -val_rmse, -test_rmse
        
#         elif args.metric == 'qwk':
#             train_qwk = cohen_kappa_score(y_true[train_idx].cpu().numpy(), predictions[train_idx].cpu().numpy(), weights="quadratic")
#             val_qwk = cohen_kappa_score(y_true[valid_idx].cpu().numpy(), predictions[valid_idx].cpu().numpy(), weights="quadratic")
#             test_qwk = cohen_kappa_score(y_true[test_idx].cpu().numpy(), predictions[test_idx].cpu().numpy(), weights="quadratic")
#             train_metric, val_metric, test_metric = train_qwk, val_qwk, test_qwk
        
#         elif args.metric == 'spearman':
#             train_spearman = spearmanr(y_true[train_idx].cpu().numpy(), predictions[train_idx].cpu().numpy()).correlation
#             val_spearman = spearmanr(y_true[valid_idx].cpu().numpy(), predictions[valid_idx].cpu().numpy()).correlation
#             test_spearman = spearmanr(y_true[test_idx].cpu().numpy(), predictions[test_idx].cpu().numpy()).correlation
#             train_spearman = 0.0 if train_spearman is None or np.isnan(train_spearman) else train_spearman
#             val_spearman = 0.0 if val_spearman is None or np.isnan(val_spearman) else val_spearman
#             test_spearman = 0.0 if test_spearman is None or np.isnan(test_spearman) else test_spearman
#             train_metric, val_metric, test_metric = train_spearman, val_spearman, test_spearman
        
#         else:
#             print(f"Warning: Unknown metric '{args.metric}'. Defaulting to accuracy.")
#             train_metric = (predictions[train_idx] == y_true[train_idx]).float().mean().item()
#             val_metric = (predictions[valid_idx] == y_true[valid_idx]).float().mean().item()
#             test_metric = (predictions[test_idx] == y_true[test_idx]).float().mean().item()
            
#         # Compute loss
#         train_loss = criterion(out[train_idx], y_true[train_idx]).item()
    
#     return train_metric, val_metric, test_metric, train_loss
# ### ----------------------------- ###


# # =====================================================================================
# #  Section 5: Main Training and Evaluation Function
# # =====================================================================================

# def train_and_evaluate(args):
#     """Main function to train and evaluate the NodeFormer model."""
#     fix_seed(args.seed)
#     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
#     args.device = device
#     print(f"Using device: {device}")

#     # Set model name for directory saving. This script is only for nodeformer.
#     args.gnn = 'nodeformer'

#     # --- Data Loading and Preprocessing ---
#     dataset = load_dataset(args.data_dir, args.dataset)
#     if len(dataset.label.shape) == 1:
#         dataset.label = dataset.label.unsqueeze(1)

#     if args.rand_split:
#         split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
#                          for _ in range(args.runs)]
#     else:
#         split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

#     dataset.label = dataset.label.to(device)

#     n = dataset.graph['num_nodes']
#     c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
#     d = dataset.graph['node_feat'].shape[1]

#     print(f'Dataset: {args.dataset}, Nodes: {n}, Features: {d}, Classes: {c}')
    
#     ### --- ORDINAL INTEGRATION --- ###
#     is_ordinal = args.dataset in ORDINAL_DATASETS
#     if is_ordinal:
#         print(f">>> Dataset '{args.dataset}' identified as ORDINAL <<<")
#         unique_labels = dataset.label.unique().cpu().numpy()
#         print(f"Unique labels: {unique_labels}")
#         if not np.array_equal(np.sort(unique_labels), np.arange(len(unique_labels))):
#             print("WARNING: Labels are not 0-indexed consecutive integers! Ordinal losses may fail.")
#     ### ----------------------------- ###

#     dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
#     dataset.graph['edge_index'], dataset.graph['node_feat'] = \
#         dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

#     # --- Model Initialization (Specific to NodeFormer) ---
#     print("Instantiating NodeFormerAdapted model...")
#     model = NodeFormerAdapted(
#         in_channels=d,
#         hidden_channels=args.hidden_channels,
#         out_channels=c,
#         local_layers=args.local_layers,
#         dropout=args.dropout,
#         heads=args.num_heads,
#         res=args.res,
#         ln=args.ln,
#         jk=args.jk
#     ).to(device)
    
#     # --- Loss, Optimizer, and Evaluation Setup ---
    
#     ### --- ORDINAL INTEGRATION --- ###
#     # Loss function selection
#     if is_ordinal:
#         if args.ordinal_loss == 'dwce':
#             criterion = DistanceWeightedCrossEntropy(num_classes=c, alpha=args.dwce_alpha)
#             print(f"Using DistanceWeightedCrossEntropy (alpha={args.dwce_alpha})")
#         elif args.ordinal_loss == 'emd':
#             criterion = EarthMoverDistanceLoss()
#             print(f"Using EarthMoverDistanceLoss")
#         elif args.ordinal_loss == 'coral':
#             criterion = CORALLoss()
#             print(f"Using CORALLoss")
#         else: # 'ce'
#             criterion = nn.CrossEntropyLoss()
#             print(f"Using standard CrossEntropyLoss (fallback)")
#     else:
#         # Standard classification losses (from original script)
#         if args.dataset in ('questions',):
#             criterion = nn.BCEWithLogitsLoss()
#             print("Using BCEWithLogitsLoss")
#         else:
#             criterion = nn.NLLLoss()
#             print("Using NLLLoss")
#     ### ----------------------------- ###
            
#     eval_func = {'prauc': eval_rocauc, 'rocauc': eval_rocauc}.get(args.metric, eval_acc)
#     logger = Logger(args.runs, args)
    
#     all_acc, all_roc_auc, all_pr_auc, all_reports = [], [], [], []
    
#     ### --- ORDINAL INTEGRATION --- ###
#     ordinal_metric_keys = ['mae', 'rmse', 'spearman', 'qwk', 'acc_exact', 'acc1', 'acc2']
#     metric_lists = {k: [] for k in ordinal_metric_keys}
#     ### ----------------------------- ###

#     exp_name = f"l_{args.local_layers}-h_{args.hidden_channels}-d_{args.dropout}-heads_{args.num_heads}"
    
#     ### --- ORDINAL INTEGRATION --- ###
#     # Add ordinal params to experiment name if used
#     if is_ordinal:
#         exp_name += f"-ord_{args.ordinal_loss}"
#         if args.ordinal_loss == 'dwce':
#             exp_name += f"-a_{args.dwce_alpha}"
#     ### ----------------------------- ###


#     # --- Training & Evaluation Loop ---
#     for run in range(args.runs):
#         print(f"\n{'='*80}")
#         print(f"Run {run+1}/{args.runs}")
#         print(f"{'='*80}")
        
#         split_idx = split_idx_lst[0] if args.dataset in ('cora', 'citeseer', 'pubmed') else split_idx_lst[run]
#         train_idx = split_idx['train'].to(device)
        
#         model.reset_parameters()
#         optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
#         best_val, best_test = float('-inf'), float('-inf')
#         best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
#         ### --- ORDINAL INTEGRATION --- ###
#         patience_counter = 0
#         ### ----------------------------- ###

#         for epoch in range(args.epochs):
#             model.train()
#             optimizer.zero_grad()
            
#             # Get raw logits from the model
#             out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

#             ### --- ORDINAL INTEGRATION --- ###
#             # Modified Loss Calculation
#             if is_ordinal:
#                 targets_train = dataset.label.squeeze(1)[train_idx].long()
#                 loss = criterion(out[train_idx], targets_train) # Pass raw logits
#             elif args.dataset in ('questions'):
#                 # Convert labels to one-hot format
#                 if dataset.label.shape[1] == 1:
#                     true_label = F.one_hot(dataset.label, num_classes=c).squeeze(1)
#                 else:
#                     true_label = dataset.label
#                 loss = criterion(out[train_idx], true_label[train_idx].to(torch.float))
#             else:
#                 # For all other datasets, apply log_softmax for NLLLoss
#                 out = F.log_softmax(out, dim=1)
#                 loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
#             ### ----------------------------- ###
            
#             loss.backward()
#             optimizer.step()

#             ### --- ORDINAL INTEGRATION --- ###
#             # Conditional Evaluation
#             if is_ordinal:
#                 result = evaluate_ordinal(model, dataset, split_idx, criterion, args)
#             else:
#                 result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
#             ### ----------------------------- ###
            
#             logger.add_result(run, result)

#             if result[1] > best_val:
#                 best_val, best_test = result[1], result[2]
#                 best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#                 patience_counter = 0
#             else:
#                 patience_counter += 1

#             if epoch % args.display_step == 0:
#                 print(f'Run: {run+1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100*result[0]:.2f}%, Valid: {100*result[1]:.2f}%, Test: {100*result[2]:.2f}%')
            
#             ### --- ORDINAL INTEGRATION --- ###
#             if args.early_stopping > 0 and patience_counter >= args.early_stopping:
#                 print(f"Early stopping at epoch {epoch}")
#                 break
#             ### ----------------------------- ###
            
#         print(f'Run {run+1:02d} finished. Best Valid: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')
#         model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
#         # --- Final Evaluation, Visualization, and Saving ---
#         model.eval()
#         with torch.no_grad():
#             out = model(dataset.graph['node_feat'], dataset.graph['edge_index']) # Raw logits

#         test_idx = split_idx['test'].to(device)
#         y_true_test = dataset.label[test_idx]
        
#         ### --- ORDINAL INTEGRATION --- ###
#         # Conditional Final Evaluation
#         ordinal_metrics = {}
        
#         if is_ordinal:
#             y_true_np = y_true_test.cpu().numpy().squeeze()
#             y_pred_tensor = get_ordinal_predictions(out[test_idx], method='expected_value')
#             y_pred_np = y_pred_tensor.cpu().numpy()
            
#             # Compute ordinal metrics
#             ordinal_metrics = compute_ordinal_metrics(y_true_np, y_pred_np)
            
#             # Use exact accuracy for 'accuracy'
#             accuracy = ordinal_metrics['acc_exact']
#             report = classification_report(y_true_np, y_pred_np, output_dict=True, zero_division=0)
            
#             # Compute ROC/PR on ordinal as multiclass
#             probs = F.softmax(out[test_idx], dim=1)
#             try:
#                 roc_auc = roc_auc_score(y_true_np, probs.cpu().numpy(), multi_class='ovr')
#                 y_true_binarized = label_binarize(y_true_np, classes=range(c))
#                 pr_auc = average_precision_score(y_true_binarized, probs.cpu().numpy())
#             except Exception:
#                 roc_auc = 0.0
#                 pr_auc = 0.0

#             print(f'\nFinal Test Accuracy (Exact): {100*accuracy:.2f}%')
#             print(f'AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}')
#             print("--- Ordinal Metrics ---")
#             print(f"MAE: {ordinal_metrics['mae']:.4f}")
#             print(f"RMSE: {ordinal_metrics['rmse']:.4f}")
#             print(f"Spearman: {ordinal_metrics['spearman']:.4f}")
#             print(f"QWK: {ordinal_metrics['qwk']:.4f}")

#         else:
#             # Standard classification (original logic)
#             out = F.log_softmax(out, dim=1) # Apply log_softmax for standard eval
#             probs = torch.exp(out[test_idx])
#             y_pred = out.argmax(dim=-1, keepdim=True)
            
#             accuracy = eval_acc(y_true_test, out[test_idx])
#             report = classification_report(y_true_test.cpu().numpy(), y_pred[test_idx].cpu().numpy(), output_dict=True, zero_division=0)
            
#             if c > 2:
#                 roc_auc = roc_auc_score(y_true_test.cpu().numpy(), probs.cpu().numpy(), multi_class='ovr')
#                 y_true_binarized = label_binarize(y_true_test.cpu().numpy(), classes=range(c))
#                 pr_auc = average_precision_score(y_true_binarized, probs.cpu().numpy())
#             else: # Binary
#                 roc_auc = eval_rocauc(y_true_test, out[test_idx])
#                 pr_auc = average_precision_score(y_true_test.cpu().numpy(), probs[:,1].cpu().numpy())

#             print(f'Final Test Accuracy: {accuracy:.4f}, AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}')
#         ### ----------------------------- ###
            
#         all_acc.append(accuracy); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc); all_reports.append(report)
        
#         ### --- ORDINAL INTEGRATION --- ###
#         if ordinal_metrics:
#             for k, v in ordinal_metrics.items():
#                 metric_lists[k].append(v)
#         ### ----------------------------- ###

#         with torch.no_grad():
#             embeddings = model.get_embeddings(dataset.graph['node_feat'], dataset.graph['edge_index'])
#             embeddings_cpu = embeddings[test_idx].cpu().numpy()
#             labels_cpu = y_true_test.cpu().numpy().squeeze()
#             tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(labels_cpu)-1))
#             tsne_embeds = tsne.fit_transform(embeddings_cpu)

#         out_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/run_{run+1}'
        
#         # Pass raw logits `out[test_idx]` for saving, regardless of type
#         save_results(model, embeddings[test_idx], tsne_embeds, labels_cpu, out[test_idx], y_true_test, args, out_dir)
#         plot_tsne(tsne_embeds, labels_cpu, out_dir)
#         plot_logits_distribution(out[test_idx], y_true_test, out_dir)

#         metrics = {'accuracy': accuracy, 'auc_roc': roc_auc, 'auc_pr': pr_auc, 'classification_report': report}
        
#         ### --- ORDINAL INTEGRATION --- ###
#         if ordinal_metrics:
#             metrics.update({f'ordinal_{k}': v for k, v in ordinal_metrics.items()})
#         ### ----------------------------- ###
            
#         with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
#             json.dump(metrics, f, indent=2)

#     # --- Summary Over All Runs ---
#     summary = {
#         'accuracy_mean': float(np.mean(all_acc)), 'accuracy_std': float(np.std(all_acc)),
#         'auc_roc_mean': float(np.mean(all_roc_auc)), 'auc_roc_std': float(np.std(all_roc_auc)),
#         'auc_pr_mean': float(np.mean(all_pr_auc)), 'auc_pr_std': float(np.std(all_pr_auc)),
#     }
    
#     ### --- ORDINAL INTEGRATION --- ###
#     if is_ordinal:
#         print("\n--- Aggregated Ordinal Metrics (mean ± std) ---")
#         for k in ordinal_metric_keys:
#             values = metric_lists[k]
#             if values:
#                 mean_val = float(np.mean(values))
#                 std_val = float(np.std(values))
#                 summary[f'ordinal_{k}_mean'] = mean_val
#                 summary[f'ordinal_{k}_std'] = std_val
#                 print(f"ordinal_{k}: {mean_val:.4f} ± {std_val:.4f}")
#     ### ----------------------------- ###
    
#     summary_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/'
#     with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)
#     print(f"\nSummary over {args.runs} runs:\n", json.dumps(summary, indent=2))
#     print(f"\nResults saved to: {summary_dir}")


# # =====================================================================================
# #  Section 6: Script Entry Point
# # =====================================================================================

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Dedicated Training Pipeline for NodeFormer')
#     parser_add_main_args(parser)
    
#     ### --- ORDINAL INTEGRATION --- ###
#     # Ordinal-specific arguments
#     parser.add_argument('--ordinal_loss', type=str, default='coral', 
#                         choices=['coral', 'dwce', 'emd', 'ce'],
#                         help="Ordinal loss: coral/dwce/emd/ce (ce=standard cross-entropy for debugging)")
#     parser.add_argument('--dwce_alpha', type=float, default=0.5,
#                         help="Alpha for DistanceWeightedCrossEntropy (default: 0.5)")
#     parser.add_argument('--early_stopping', type=int, default=0,
#                         help="Early stopping patience (0 = disabled)")
#     ### ----------------------------- ###
    
#     args = parser.parse_args()
    
#     # Print configuration
#     print("\n" + "="*80)
#     print("CONFIGURATION")
#     print("="*80)
#     print(f"Dataset: {args.dataset}")
#     print(f"Model: NodeFormer")
#     print(f"Metric for Val: {args.metric}")
#     if args.dataset in ORDINAL_DATASETS:
#         print(f"Ordinal Loss: {args.ordinal_loss}")
#         if args.ordinal_loss == 'dwce':
#             print(f"  - DWCE Alpha: {args.dwce_alpha}")
#     print(f"Learning Rate: {args.lr}")
#     print(f"Weight Decay: {args.weight_decay}")
#     print(f"Hidden Channels: {args.hidden_channels}")
#     print(f"Dropout: {args.dropout}")
#     print(f"Epochs: {args.epochs}")
#     print(f"Early Stopping: {args.early_stopping}")
#     print(f"Runs: {args.runs}")
#     print(f"Seed: {args.seed}")
#     print("="*80 + "\n")
    
#     train_and_evaluate(args)