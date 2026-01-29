# =====================================================================================
#  Fixed Integrated GNN Training Pipeline with Corrected Ordinal Losses
# =====================================================================================

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, to_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, balanced_accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
import json
import scipy.sparse as sp
import networkx as nx

# --- Model Imports ---
from model import MPNNs
from mlp_model import MLP

# --- Helper Imports ---
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, eval_prauc, class_rand_splits, load_fixed_splits, eval_balanced_acc
from eval import evaluate
from logger import Logger
from parse import parser_add_main_args

# =====================================================================================
#  Section 0: Configuration
# =====================================================================================
ORDINAL_DATASETS = ['amazon-ratings', 'squirrel']

# =====================================================================================
#  Section 1: Helper Functions
# =====================================================================================

def fix_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_tensor_sparse(mx, symmetric=0):
    """Row-normalize or symmetrically-normalize a sparse matrix."""
    rowsum = np.array(mx.sum(1)) + 1e-12
    if symmetric == 0:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        return r_mat_inv.dot(mx)
    else:
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        mx = r_mat_inv.dot(mx)
        return mx.dot(r_mat_inv)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def save_results(model, tsne_labels, logits, labels, args, out_dir):
    """Saves all run artifacts to a directory."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
    np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
    np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

def generate_param_string(args):
    """Generates a unique, readable string from all relevant hyperparameters."""
    params = []
    params.append(f"metric-{args.metric}")
    params.append(f"lr-{args.lr}")
    params.append(f"wd-{args.weight_decay}")
    params.append(f"do-{args.dropout}")
    params.append(f"hid-{args.hidden_channels}")
    gnn_type = args.gnn
    if gnn_type in ['gcn', 'gat', 'sage', 'gin']:
        params.append(f"layers-{args.local_layers}")
        if gnn_type == 'gat':
            params.append(f"heads-{args.num_heads}")
        if args.res: params.append('res')
        if args.ln: params.append('ln')
        if args.bn: params.append('bn')
        if args.jk: params.append('jk')
    if getattr(args, 'ordinal_loss', None):
        params.append(f"ordinal-{args.ordinal_loss}")
        if args.ordinal_loss == 'dwce':
            params.append(f"alpha-{args.dwce_alpha}")
    return "_".join(params)

# =====================================================================================
#  Section 2: CORRECTED Ordinal Loss Implementations
# =====================================================================================

class DistanceWeightedCrossEntropy(nn.Module):
    """
    Distance-weighted cross-entropy.
    Penalizes misclassifications proportional to their distance from true class.
    """
    def __init__(self, num_classes, alpha=1.0, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        # Distance matrix: dist[i,j] = |i - j|
        dist = torch.abs(torch.arange(num_classes).float().unsqueeze(1) - 
                        torch.arange(num_classes).float().unsqueeze(0))
        # Weight increases with distance: w[i,j] = 1 + alpha * |i - j|
        self.register_buffer('weight_mat', 1.0 + self.alpha * dist)

    def forward(self, logits, targets):
        """
        logits: [N, C] raw logits
        targets: [N] class indices
        """
        device = logits.device
        # Ensure weight_mat is on correct device
        if self.weight_mat.device != device:
            self.weight_mat = self.weight_mat.to(device)
        
        # Standard cross-entropy loss per sample
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')  # [N]
        
        # Get predicted class
        pred_class = logits.argmax(dim=1)  # [N]
        
        # Get distance weights for (true_class, pred_class) pairs
        weights = self.weight_mat[targets, pred_class]  # [N]
        
        # Apply distance weighting
        weighted_loss = ce_loss * weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class EarthMoverDistanceLoss(nn.Module):
    """
    EMD-like loss using cumulative distribution difference.
    Measures L1 distance between cumulative distributions.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [N, C] raw logits
        targets: [N] class indices
        """
        N, C = logits.shape
        device = logits.device
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # [N, C]
        
        # Compute cumulative probabilities
        cum_probs = torch.cumsum(probs, dim=1)  # [N, C]
        
        # Create cumulative target distribution
        # For true class k: [0, 0, ..., 0, 1, 1, ..., 1]
        arange = torch.arange(C, device=device).unsqueeze(0).expand(N, -1)  # [N, C]
        targets_expanded = targets.unsqueeze(1)  # [N, 1]
        cum_target = (arange >= targets_expanded).float()  # [N, C]
        
        # Compute L1 distance between cumulative distributions
        diff = torch.abs(cum_probs - cum_target)
        per_sample_loss = diff.sum(dim=1)  # Sum over classes
        
        if self.reduction == 'mean':
            return per_sample_loss.mean()
        elif self.reduction == 'sum':
            return per_sample_loss.sum()
        else:
            return per_sample_loss


class CORALLoss(nn.Module):
    """
    Simplified CORAL (Consistent Rank Logits) loss.
    Uses standard softmax outputs and converts to cumulative probabilities.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [N, C] raw logits
        targets: [N] class indices in range [0, C-1]
        """
        N, C = logits.shape
        device = logits.device
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # [N, C]
        
        # Compute cumulative probabilities for each threshold k
        # P(Y > k) = sum of probabilities for classes > k
        cum_probs = torch.zeros(N, C-1, device=device)
        for k in range(C-1):
            # P(Y > k) = sum_{j=k+1}^{C-1} p_j
            cum_probs[:, k] = probs[:, k+1:].sum(dim=1)
        
        # Target: P(Y > k) should be 1 if true_class > k, else 0
        target_cum = torch.zeros(N, C-1, device=device)
        for k in range(C-1):
            target_cum[:, k] = (targets > k).float()
        
        # Binary cross-entropy loss for each threshold
        # Clamp probabilities to avoid log(0)
        cum_probs = torch.clamp(cum_probs, min=1e-7, max=1-1e-7)
        
        bce_loss = -(target_cum * torch.log(cum_probs) + 
                     (1 - target_cum) * torch.log(1 - cum_probs))
        
        per_sample_loss = bce_loss.sum(dim=1)  # Sum over thresholds
        
        if self.reduction == 'mean':
            return per_sample_loss.mean()
        elif self.reduction == 'sum':
            return per_sample_loss.sum()
        else:
            return per_sample_loss


# =====================================================================================
#  Section 3: Ordinal Prediction and Metrics
# =====================================================================================

def get_ordinal_predictions(logits, method='expected_value'):
    """
    Convert logits to ordinal predictions.
    
    Args:
        logits: [N, C] raw logits
        method: 'argmax' or 'expected_value'
    
    Returns:
        predictions: [N] predicted class indices
    """
    probs = F.softmax(logits, dim=1)  # [N, C]
    
    if method == 'argmax':
        return probs.argmax(dim=1)
    elif method == 'expected_value':
        # E[Y] = sum_i i * P(Y=i)
        C = logits.shape[1]
        class_indices = torch.arange(C, device=logits.device).float()
        pred = (probs * class_indices).sum(dim=1)  # [N]
        # Round to nearest integer
        return pred.round().long()
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_ordinal_metrics(y_true_tensor, y_pred_tensor):
    """Compute comprehensive ordinal metrics."""
    y_true = y_true_tensor.cpu().numpy() if torch.is_tensor(y_true_tensor) else y_true_tensor
    y_pred = y_pred_tensor.cpu().numpy() if torch.is_tensor(y_pred_tensor) else y_pred_tensor
    
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Spearman correlation
    try:
        spearman_corr = spearmanr(y_true, y_pred).correlation
        if spearman_corr is None or np.isnan(spearman_corr):
            spearman_corr = 0.0
    except Exception:
        spearman_corr = 0.0
    
    # Quadratic Weighted Kappa
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        qwk = 0.0
    
    # Off-by-k accuracy
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

def evaluate_ordinal(model, dataset, split_idx, criterion, args, 
                     fsgcn_list_mat=None, glognn_features=None, 
                     glognn_adj_sparse=None, glognn_adj_dense=None,
                     gprgnn_features=None):
    """
    Evaluate function specifically for ordinal datasets.
    Returns metric based on args.metric (acc, mae, rmse, qwk, spearman)
    Returns: (train_metric, val_metric, test_metric, train_loss)
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        if args.gnn == 'fsgcn':
            out = model(fsgcn_list_mat)
        elif args.gnn == 'glognn':
            out = model(glognn_features, glognn_adj_sparse, glognn_adj_dense)
        elif args.gnn == 'gprgnn':
            out = model(gprgnn_features, dataset.graph['edge_index'])
        elif args.gnn == 'mlp':
            out = model(dataset.graph['node_feat'])
        else:
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        
        # Get predictions using expected value for ordinal
        predictions = get_ordinal_predictions(out, method='expected_value')
        
        # Get true labels
        y_true = dataset.label.squeeze(1)
        
        # Compute metric for each split based on args.metric
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        
        if args.metric == 'acc':
            # Standard accuracy
            train_metric = (predictions[train_idx] == y_true[train_idx]).float().mean().item()
            val_metric = (predictions[valid_idx] == y_true[valid_idx]).float().mean().item()
            test_metric = (predictions[test_idx] == y_true[test_idx]).float().mean().item()
        
        elif args.metric == 'mae':
            # Mean Absolute Error (lower is better, so return negative for maximization)
            train_mae = torch.abs(predictions[train_idx].float() - y_true[train_idx].float()).mean().item()
            val_mae = torch.abs(predictions[valid_idx].float() - y_true[valid_idx].float()).mean().item()
            test_mae = torch.abs(predictions[test_idx].float() - y_true[test_idx].float()).mean().item()
            train_metric, val_metric, test_metric = -train_mae, -val_mae, -test_mae
        
        elif args.metric == 'rmse':
            # Root Mean Squared Error (lower is better, so return negative for maximization)
            train_rmse = torch.sqrt(((predictions[train_idx].float() - y_true[train_idx].float()) ** 2).mean()).item()
            val_rmse = torch.sqrt(((predictions[valid_idx].float() - y_true[valid_idx].float()) ** 2).mean()).item()
            test_rmse = torch.sqrt(((predictions[test_idx].float() - y_true[test_idx].float()) ** 2).mean()).item()
            train_metric, val_metric, test_metric = -train_rmse, -val_rmse, -test_rmse
        
        elif args.metric == 'qwk':
            # Quadratic Weighted Kappa
            train_qwk = cohen_kappa_score(
                y_true[train_idx].cpu().numpy(), 
                predictions[train_idx].cpu().numpy(), 
                weights="quadratic"
            )
            val_qwk = cohen_kappa_score(
                y_true[valid_idx].cpu().numpy(), 
                predictions[valid_idx].cpu().numpy(), 
                weights="quadratic"
            )
            test_qwk = cohen_kappa_score(
                y_true[test_idx].cpu().numpy(), 
                predictions[test_idx].cpu().numpy(), 
                weights="quadratic"
            )
            train_metric, val_metric, test_metric = train_qwk, val_qwk, test_qwk
        
        elif args.metric == 'spearman':
            # Spearman correlation
            train_spearman = spearmanr(
                y_true[train_idx].cpu().numpy(), 
                predictions[train_idx].cpu().numpy()
            ).correlation
            val_spearman = spearmanr(
                y_true[valid_idx].cpu().numpy(), 
                predictions[valid_idx].cpu().numpy()
            ).correlation
            test_spearman = spearmanr(
                y_true[test_idx].cpu().numpy(), 
                predictions[test_idx].cpu().numpy()
            ).correlation
            # Handle NaN cases
            train_spearman = 0.0 if train_spearman is None or np.isnan(train_spearman) else train_spearman
            val_spearman = 0.0 if val_spearman is None or np.isnan(val_spearman) else val_spearman
            test_spearman = 0.0 if test_spearman is None or np.isnan(test_spearman) else test_spearman
            train_metric, val_metric, test_metric = train_spearman, val_spearman, test_spearman
        
        else:
            # Default to accuracy
            train_metric = (predictions[train_idx] == y_true[train_idx]).float().mean().item()
            val_metric = (predictions[valid_idx] == y_true[valid_idx]).float().mean().item()
            test_metric = (predictions[test_idx] == y_true[test_idx]).float().mean().item()
        
        # Compute loss
        train_loss = criterion(out[train_idx], y_true[train_idx]).item()
    
    return train_metric, val_metric, test_metric, train_loss

# =====================================================================================
#  Section 5: Main Training and Evaluation Function
# =====================================================================================

def train_and_evaluate(args):
    """Main function to run the training and evaluation pipeline."""
    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Device: {device}')

    # Load dataset
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    # Get splits
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) 
                        for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    
    print(f'Dataset: {args.dataset}, Nodes: {n}, Classes: {c}, Features: {d}')
    
    # Check if dataset is ordinal
    is_ordinal = args.dataset in ORDINAL_DATASETS
    if is_ordinal:
        print(f">>> Dataset '{args.dataset}' identified as ORDINAL <<<")
        # Verify labels are 0-indexed integers
        unique_labels = dataset.label.unique().cpu().numpy()
        print(f"Unique labels: {unique_labels}")
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            print("WARNING: Labels are not 0-indexed consecutive integers!")
    
    # Pre-computation (simplified - keeping only standard path)
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    
    # Model initialization
    if args.gnn == 'mlp':
        model = MLP(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout).to(device)
    else:
        model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, 
                     dropout=args.dropout, heads=args.num_heads, 
                     pre_ln=args.pre_ln, pre_linear=args.pre_linear,
                     res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, 
                     gnn=args.gnn).to(device)
    
    print(f"Model: {args.gnn.upper()}, Parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss function selection
    if is_ordinal:
        if args.ordinal_loss == 'dwce':
            criterion = DistanceWeightedCrossEntropy(num_classes=c, alpha=args.dwce_alpha)
            print(f"Using DistanceWeightedCrossEntropy (alpha={args.dwce_alpha})")
        elif args.ordinal_loss == 'emd':
            criterion = EarthMoverDistanceLoss()
            print(f"Using EarthMoverDistanceLoss")
        elif args.ordinal_loss == 'coral':
            criterion = CORALLoss()
            print(f"Using CORALLoss")
        else:
            # Fallback to standard CE for debugging
            criterion = nn.CrossEntropyLoss()
            print(f"Using standard CrossEntropyLoss (fallback)")
    else:
        if args.dataset in ('questions',):
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

    # Storage for results
    all_results = []

    for run in range(args.runs):
        print(f"\n{'='*80}")
        print(f"Run {run+1}/{args.runs}")
        print(f"{'='*80}")
        
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        valid_idx = split_idx['valid'].to(device)
        test_idx = split_idx['test'].to(device)
        
        print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
        
        # Reset model
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), 
                                    weight_decay=args.weight_decay, 
                                    lr=args.lr)

        best_val = float('-inf')
        best_test = float('-inf')
        best_epoch = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            # Forward pass
            if args.gnn == 'mlp':
                out = model(dataset.graph['node_feat'])
            else:
                out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            # Compute loss
            if is_ordinal:
                targets_train = dataset.label.squeeze(1)[train_idx].long()
                loss = criterion(out[train_idx], targets_train)
            else:
                if args.dataset in ('questions',):
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1) \
                                 if dataset.label.shape[1] == 1 else dataset.label
                    loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
                else:
                    loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

            loss.backward()
            optimizer.step()

            # Evaluation
            if is_ordinal:
                result = evaluate_ordinal(model, dataset, split_idx, criterion, args)
            else:
                eval_func = {'prauc': eval_prauc, 'rocauc': eval_rocauc, 
                           'balacc': eval_balanced_acc}.get(args.metric, eval_acc)
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            
            train_metric, val_metric, test_metric, train_loss = result

            # Track best model
            if val_metric > best_val:
                best_val = val_metric
                best_test = test_metric
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Display progress
            if epoch % args.display_step == 0 or epoch == args.epochs - 1:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {100*train_metric:.2f}%, '
                      f'Val: {100*val_metric:.2f}%, '
                      f'Test: {100*test_metric:.2f}%')
            
            # Early stopping
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f'\nBest Epoch: {best_epoch}, Best Val: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')

        # Load best model for final evaluation
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()
        
        with torch.no_grad():
            if args.gnn == 'mlp':
                out = model(dataset.graph['node_feat'])
            else:
                out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

        # Final evaluation metrics
        y_true = dataset.label.squeeze(1)
        
        # Standard predictions
        if is_ordinal:
            y_pred = get_ordinal_predictions(out, method='expected_value')
        else:
            y_pred = out.argmax(dim=-1)

        # Compute metrics on test set
        test_acc = (y_pred[test_idx] == y_true[test_idx]).float().mean().item()
        
        if c > 2:
            probs = F.softmax(out[test_idx], dim=1)
            try:
                roc_auc = roc_auc_score(y_true[test_idx].cpu().numpy(), 
                                       probs.cpu().numpy(), 
                                       multi_class='ovr')
            except:
                roc_auc = 0.0
            balanced_acc = balanced_accuracy_score(y_true[test_idx].cpu().numpy(), 
                                                  y_pred[test_idx].cpu().numpy())
        else:
            probs = F.softmax(out[test_idx], dim=1)
            try:
                roc_auc = roc_auc_score(y_true[test_idx].cpu().numpy(), 
                                       probs[:, 1].cpu().numpy())
            except:
                roc_auc = 0.0
            balanced_acc = balanced_accuracy_score(y_true[test_idx].cpu().numpy(), 
                                                  y_pred[test_idx].cpu().numpy())

        print(f'\nTest Accuracy: {100*test_acc:.2f}%')
        print(f'Balanced Accuracy: {100*balanced_acc:.2f}%')
        print(f'ROC-AUC: {roc_auc:.4f}')

        # Ordinal-specific metrics
        ordinal_metrics = {}
        if is_ordinal:
            ordinal_metrics = compute_ordinal_metrics(
                y_true[test_idx].cpu().numpy(),
                y_pred[test_idx].cpu().numpy()
            )
            print("\n--- Ordinal Metrics ---")
            print(f"MAE: {ordinal_metrics['mae']:.4f}")
            print(f"RMSE: {ordinal_metrics['rmse']:.4f}")
            print(f"Spearman: {ordinal_metrics['spearman']:.4f}")
            print(f"QWK: {ordinal_metrics['qwk']:.4f}")
            print(f"Acc@1: {100*ordinal_metrics['acc1']:.2f}%")
            print(f"Acc@2: {100*ordinal_metrics['acc2']:.2f}%")

        # Save results
        run_results = {
            'run': run,
            'best_epoch': best_epoch,
            'best_val': best_val,
            'best_test': best_test,
            'test_accuracy': test_acc,
            'balanced_accuracy': balanced_acc,
            'roc_auc': roc_auc,
        }
        
        if ordinal_metrics:
            run_results.update({
                'ordinal_mae': ordinal_metrics['mae'],
                'ordinal_rmse': ordinal_metrics['rmse'],
                'ordinal_spearman': ordinal_metrics['spearman'],
                'ordinal_qwk': ordinal_metrics['qwk'],
                'ordinal_acc1': ordinal_metrics['acc1'],
                'ordinal_acc2': ordinal_metrics['acc2'],
            })
        
        all_results.append(run_results)

        # Save model and predictions
        param_string = generate_param_string(args)
        out_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/run_{run}'
        save_results(model, y_true[test_idx].cpu().numpy(), 
                    out[test_idx], y_true[test_idx], args, out_dir)
        
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(run_results, f, indent=2)

    # Aggregate results
    print(f"\n{'='*80}")
    print("FINAL RESULTS (mean ± std)")
    print(f"{'='*80}")
    
    metrics_to_aggregate = ['best_val', 'best_test', 'test_accuracy', 
                           'balanced_accuracy', 'roc_auc']
    if is_ordinal:
        metrics_to_aggregate.extend(['ordinal_mae', 'ordinal_rmse', 
                                     'ordinal_spearman', 'ordinal_qwk', 
                                     'ordinal_acc1', 'ordinal_acc2'])
    
    summary = {}
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in all_results if metric in r]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[f'{metric}_mean'] = float(mean_val)
            summary[f'{metric}_std'] = float(std_val)
            
            # Pretty print
            if 'acc' in metric.lower():
                print(f"{metric}: {100*mean_val:.2f} ± {100*std_val:.2f}%")
            else:
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    # Save summary
    param_string = generate_param_string(args)
    summary_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/summary'
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(os.path.join(summary_dir, 'all_runs.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {summary_dir}")

# =====================================================================================
#  Section 6: Script Entry Point
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fixed GNN Training Pipeline with Ordinal Losses')
    parser_add_main_args(parser)

    # Ordinal-specific arguments
    parser.add_argument('--ordinal_loss', type=str, default='coral', 
                        choices=['coral', 'dwce', 'emd', 'ce'],
                        help="Ordinal loss: coral/dwce/emd/ce (ce=standard cross-entropy for debugging)")
    parser.add_argument('--dwce_alpha', type=float, default=0.01,
                        help="Alpha for DistanceWeightedCrossEntropy (default: 0.5, lower = less penalty)")
    parser.add_argument('--early_stopping', type=int, default=0,
                        help="Early stopping patience (0 = disabled)")

    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.gnn.upper()}")
    print(f"Ordinal Loss: {args.ordinal_loss}")
    if args.ordinal_loss == 'dwce':
        print(f"  - Alpha: {args.dwce_alpha}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Hidden Channels: {args.hidden_channels}")
    print(f"Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Runs: {args.runs}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")
    
    train_and_evaluate(args)




# # =====================================================================================
# #  Fixed Integrated GNN Training Pipeline with Corrected Ordinal Losses
# # =====================================================================================

# import argparse
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, to_scipy_sparse_matrix
# from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, balanced_accuracy_score, f1_score
# from sklearn.metrics import cohen_kappa_score
# from scipy.stats import spearmanr
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import os
# import sys
# import json
# import scipy.sparse as sp
# import networkx as nx

# # --- Model Imports ---
# from model import MPNNs
# from mlp_model import MLP

# # --- Helper Imports ---
# from dataset import load_dataset
# from data_utils import eval_acc, eval_rocauc, eval_prauc, class_rand_splits, load_fixed_splits, eval_balanced_acc
# from eval import evaluate
# from logger import Logger
# from parse import parser_add_main_args

# # =====================================================================================
# #  Section 0: Configuration
# # =====================================================================================
# ORDINAL_DATASETS = ['amazon-ratings', 'squirrel']

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

# def normalize_tensor_sparse(mx, symmetric=0):
#     """Row-normalize or symmetrically-normalize a sparse matrix."""
#     rowsum = np.array(mx.sum(1)) + 1e-12
#     if symmetric == 0:
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv, 0)
#         return r_mat_inv.dot(mx)
#     else:
#         r_inv = np.power(rowsum, -0.5).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv, 0)
#         mx = r_mat_inv.dot(mx)
#         return mx.dot(r_mat_inv)

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)

# def save_results(model, tsne_labels, logits, labels, args, out_dir):
#     """Saves all run artifacts to a directory."""
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
#     np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
#     np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
#     np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

# def generate_param_string(args):
#     """Generates a unique, readable string from all relevant hyperparameters."""
#     params = []
#     params.append(f"metric-{args.metric}")
#     params.append(f"lr-{args.lr}")
#     params.append(f"wd-{args.weight_decay}")
#     params.append(f"do-{args.dropout}")
#     params.append(f"hid-{args.hidden_channels}")
#     gnn_type = args.gnn
#     if gnn_type in ['gcn', 'gat', 'sage', 'gin']:
#         params.append(f"layers-{args.local_layers}")
#         if gnn_type == 'gat':
#             params.append(f"heads-{args.num_heads}")
#         if args.res: params.append('res')
#         if args.ln: params.append('ln')
#         if args.bn: params.append('bn')
#         if args.jk: params.append('jk')
#     if getattr(args, 'ordinal_loss', None):
#         params.append(f"ordinal-{args.ordinal_loss}")
#         if args.ordinal_loss == 'dwce':
#             params.append(f"alpha-{args.dwce_alpha}")
#     return "_".join(params)

# # =====================================================================================
# #  Section 2: CORRECTED Ordinal Loss Implementations
# # =====================================================================================

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
#                         torch.arange(num_classes).float().unsqueeze(0))
#         # Weight increases with distance: w[i,j] = 1 + alpha * |i - j|
#         self.register_buffer('weight_mat', 1.0 + self.alpha * dist)

#     def forward(self, logits, targets):
#         """
#         logits: [N, C] raw logits
#         targets: [N] class indices
#         """
#         device = logits.device
#         # Ensure weight_mat is on correct device
#         if self.weight_mat.device != device:
#             self.weight_mat = self.weight_mat.to(device)
        
#         # Standard cross-entropy loss per sample
#         log_probs = F.log_softmax(logits, dim=1)
#         ce_loss = F.nll_loss(log_probs, targets, reduction='none')  # [N]
        
#         # Get predicted class
#         pred_class = logits.argmax(dim=1)  # [N]
        
#         # Get distance weights for (true_class, pred_class) pairs
#         weights = self.weight_mat[targets, pred_class]  # [N]
        
#         # Apply distance weighting
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
#     Measures L1 distance between cumulative distributions.
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
        
#         # Convert logits to probabilities
#         probs = F.softmax(logits, dim=1)  # [N, C]
        
#         # Compute cumulative probabilities
#         cum_probs = torch.cumsum(probs, dim=1)  # [N, C]
        
#         # Create cumulative target distribution
#         # For true class k: [0, 0, ..., 0, 1, 1, ..., 1]
#         arange = torch.arange(C, device=device).unsqueeze(0).expand(N, -1)  # [N, C]
#         targets_expanded = targets.unsqueeze(1)  # [N, 1]
#         cum_target = (arange >= targets_expanded).float()  # [N, C]
        
#         # Compute L1 distance between cumulative distributions
#         diff = torch.abs(cum_probs - cum_target)
#         per_sample_loss = diff.sum(dim=1)  # Sum over classes
        
#         if self.reduction == 'mean':
#             return per_sample_loss.mean()
#         elif self.reduction == 'sum':
#             return per_sample_loss.sum()
#         else:
#             return per_sample_loss


# class CORALLoss(nn.Module):
#     """
#     Simplified CORAL (Consistent Rank Logits) loss.
#     Uses standard softmax outputs and converts to cumulative probabilities.
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
        
#         # Get probabilities
#         probs = F.softmax(logits, dim=1)  # [N, C]
        
#         # Compute cumulative probabilities for each threshold k
#         # P(Y > k) = sum of probabilities for classes > k
#         cum_probs = torch.zeros(N, C-1, device=device)
#         for k in range(C-1):
#             # P(Y > k) = sum_{j=k+1}^{C-1} p_j
#             cum_probs[:, k] = probs[:, k+1:].sum(dim=1)
        
#         # Target: P(Y > k) should be 1 if true_class > k, else 0
#         target_cum = torch.zeros(N, C-1, device=device)
#         for k in range(C-1):
#             target_cum[:, k] = (targets > k).float()
        
#         # Binary cross-entropy loss for each threshold
#         # Clamp probabilities to avoid log(0)
#         cum_probs = torch.clamp(cum_probs, min=1e-7, max=1-1e-7)
        
#         bce_loss = -(target_cum * torch.log(cum_probs) + 
#                      (1 - target_cum) * torch.log(1 - cum_probs))
        
#         per_sample_loss = bce_loss.sum(dim=1)  # Sum over thresholds
        
#         if self.reduction == 'mean':
#             return per_sample_loss.mean()
#         elif self.reduction == 'sum':
#             return per_sample_loss.sum()
#         else:
#             return per_sample_loss


# # =====================================================================================
# #  Section 3: Ordinal Prediction and Metrics
# # =====================================================================================

# def get_ordinal_predictions(logits, method='expected_value'):
#     """
#     Convert logits to ordinal predictions.
    
#     Args:
#         logits: [N, C] raw logits
#         method: 'argmax' or 'expected_value'
    
#     Returns:
#         predictions: [N] predicted class indices
#     """
#     probs = F.softmax(logits, dim=1)  # [N, C]
    
#     if method == 'argmax':
#         return probs.argmax(dim=1)
#     elif method == 'expected_value':
#         # E[Y] = sum_i i * P(Y=i)
#         C = logits.shape[1]
#         class_indices = torch.arange(C, device=logits.device).float()
#         pred = (probs * class_indices).sum(dim=1)  # [N]
#         # Round to nearest integer
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
    
#     # Spearman correlation
#     try:
#         spearman_corr = spearmanr(y_true, y_pred).correlation
#         if spearman_corr is None or np.isnan(spearman_corr):
#             spearman_corr = 0.0
#     except Exception:
#         spearman_corr = 0.0
    
#     # Quadratic Weighted Kappa
#     try:
#         qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
#     except Exception:
#         qwk = 0.0
    
#     # Off-by-k accuracy
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

# # =====================================================================================
# #  Section 4: Custom Evaluate Function for Ordinal
# # =====================================================================================

# def evaluate_ordinal(model, dataset, split_idx, criterion, args, 
#                      fsgcn_list_mat=None, glognn_features=None, 
#                      glognn_adj_sparse=None, glognn_adj_dense=None,
#                      gprgnn_features=None):
#     """
#     Evaluate function specifically for ordinal datasets.
#     Returns: (train_acc, val_acc, test_acc, train_loss)
#     """
#     model.eval()
    
#     with torch.no_grad():
#         # Forward pass
#         if args.gnn == 'fsgcn':
#             out = model(fsgcn_list_mat)
#         elif args.gnn == 'glognn':
#             out = model(glognn_features, glognn_adj_sparse, glognn_adj_dense)
#         elif args.gnn == 'gprgnn':
#             out = model(gprgnn_features, dataset.graph['edge_index'])
#         elif args.gnn == 'mlp':
#             out = model(dataset.graph['node_feat'])
#         else:
#             out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        
#         # Get predictions using expected value for ordinal
#         predictions = get_ordinal_predictions(out, method='expected_value')
        
#         # Get true labels
#         y_true = dataset.label.squeeze(1)
        
#         # Compute accuracy for each split
#         train_idx = split_idx['train']
#         valid_idx = split_idx['valid']
#         test_idx = split_idx['test']
        
#         train_acc = (predictions[train_idx] == y_true[train_idx]).float().mean().item()
#         val_acc = (predictions[valid_idx] == y_true[valid_idx]).float().mean().item()
#         test_acc = (predictions[test_idx] == y_true[test_idx]).float().mean().item()
        
#         # Compute loss
#         train_loss = criterion(out[train_idx], y_true[train_idx]).item()
    
#     return train_acc, val_acc, test_acc, train_loss

# # =====================================================================================
# #  Section 5: Main Training and Evaluation Function
# # =====================================================================================

# def train_and_evaluate(args):
#     """Main function to run the training and evaluation pipeline."""
#     fix_seed(args.seed)
#     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
#     print(f'Device: {device}')

#     # Load dataset
#     dataset = load_dataset(args.data_dir, args.dataset)
#     if len(dataset.label.shape) == 1:
#         dataset.label = dataset.label.unsqueeze(1)

#     # Get splits
#     if args.rand_split:
#         split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) 
#                         for _ in range(args.runs)]
#     else:
#         split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

#     dataset.label = dataset.label.to(device)

#     n = dataset.graph['num_nodes']
#     c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
#     d = dataset.graph['node_feat'].shape[1]
    
#     print(f'Dataset: {args.dataset}, Nodes: {n}, Classes: {c}, Features: {d}')
    
#     # Check if dataset is ordinal
#     is_ordinal = args.dataset in ORDINAL_DATASETS
#     if is_ordinal:
#         print(f">>> Dataset '{args.dataset}' identified as ORDINAL <<<")
#         # Verify labels are 0-indexed integers
#         unique_labels = dataset.label.unique().cpu().numpy()
#         print(f"Unique labels: {unique_labels}")
#         if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
#             print("WARNING: Labels are not 0-indexed consecutive integers!")
    
#     # Pre-computation (simplified - keeping only standard path)
#     dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
#     dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
#     dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
#     dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    
#     # Model initialization
#     if args.gnn == 'mlp':
#         model = MLP(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout).to(device)
#     else:
#         model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, 
#                      dropout=args.dropout, heads=args.num_heads, 
#                      pre_ln=args.pre_ln, pre_linear=args.pre_linear,
#                      res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, 
#                      gnn=args.gnn).to(device)
    
#     print(f"Model: {args.gnn.upper()}, Parameters: {sum(p.numel() for p in model.parameters())}")

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
#             # Fallback to standard CE for debugging
#             criterion = nn.CrossEntropyLoss()
#             print(f"Using standard CrossEntropyLoss (fallback)")
#     else:
#         if args.dataset in ('questions',):
#             criterion = nn.BCEWithLogitsLoss()
#         else:
#             criterion = nn.CrossEntropyLoss()

#     # Storage for results
#     all_results = []

#     for run in range(args.runs):
#         print(f"\n{'='*80}")
#         print(f"Run {run+1}/{args.runs}")
#         print(f"{'='*80}")
        
#         split_idx = split_idx_lst[run]
#         train_idx = split_idx['train'].to(device)
#         valid_idx = split_idx['valid'].to(device)
#         test_idx = split_idx['test'].to(device)
        
#         print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
        
#         # Reset model
#         model.reset_parameters()
#         optimizer = torch.optim.Adam(model.parameters(), 
#                                     weight_decay=args.weight_decay, 
#                                     lr=args.lr)

#         best_val = float('-inf')
#         best_test = float('-inf')
#         best_epoch = 0
#         best_model_state = None
#         patience_counter = 0
        
#         for epoch in range(args.epochs):
#             # Training
#             model.train()
#             optimizer.zero_grad()

#             # Forward pass
#             if args.gnn == 'mlp':
#                 out = model(dataset.graph['node_feat'])
#             else:
#                 out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

#             # Compute loss
#             if is_ordinal:
#                 targets_train = dataset.label.squeeze(1)[train_idx].long()
#                 loss = criterion(out[train_idx], targets_train)
#             else:
#                 if args.dataset in ('questions',):
#                     true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1) \
#                                  if dataset.label.shape[1] == 1 else dataset.label
#                     loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
#                 else:
#                     loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

#             loss.backward()
#             optimizer.step()

#             # Evaluation
#             if is_ordinal:
#                 result = evaluate_ordinal(model, dataset, split_idx, criterion, args)
#             else:
#                 eval_func = {'prauc': eval_prauc, 'rocauc': eval_rocauc, 
#                            'balacc': eval_balanced_acc}.get(args.metric, eval_acc)
#                 result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            
#             train_metric, val_metric, test_metric, train_loss = result

#             # Track best model
#             if val_metric > best_val:
#                 best_val = val_metric
#                 best_test = test_metric
#                 best_epoch = epoch
#                 best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#                 patience_counter = 0
#             else:
#                 patience_counter += 1

#             # Display progress
#             if epoch % args.display_step == 0 or epoch == args.epochs - 1:
#                 print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#                       f'Train: {100*train_metric:.2f}%, '
#                       f'Val: {100*val_metric:.2f}%, '
#                       f'Test: {100*test_metric:.2f}%')
            
#             # Early stopping
#             if args.early_stopping > 0 and patience_counter >= args.early_stopping:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

#         print(f'\nBest Epoch: {best_epoch}, Best Val: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')

#         # Load best model for final evaluation
#         model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
#         model.eval()
        
#         with torch.no_grad():
#             if args.gnn == 'mlp':
#                 out = model(dataset.graph['node_feat'])
#             else:
#                 out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

#         # Final evaluation metrics
#         y_true = dataset.label.squeeze(1)
        
#         # Standard predictions
#         if is_ordinal:
#             y_pred = get_ordinal_predictions(out, method='expected_value')
#         else:
#             y_pred = out.argmax(dim=-1)

#         # Compute metrics on test set
#         test_acc = (y_pred[test_idx] == y_true[test_idx]).float().mean().item()
        
#         if c > 2:
#             probs = F.softmax(out[test_idx], dim=1)
#             try:
#                 roc_auc = roc_auc_score(y_true[test_idx].cpu().numpy(), 
#                                        probs.cpu().numpy(), 
#                                        multi_class='ovr')
#             except:
#                 roc_auc = 0.0
#             balanced_acc = balanced_accuracy_score(y_true[test_idx].cpu().numpy(), 
#                                                   y_pred[test_idx].cpu().numpy())
#         else:
#             probs = F.softmax(out[test_idx], dim=1)
#             try:
#                 roc_auc = roc_auc_score(y_true[test_idx].cpu().numpy(), 
#                                        probs[:, 1].cpu().numpy())
#             except:
#                 roc_auc = 0.0
#             balanced_acc = balanced_accuracy_score(y_true[test_idx].cpu().numpy(), 
#                                                   y_pred[test_idx].cpu().numpy())

#         print(f'\nTest Accuracy: {100*test_acc:.2f}%')
#         print(f'Balanced Accuracy: {100*balanced_acc:.2f}%')
#         print(f'ROC-AUC: {roc_auc:.4f}')

#         # Ordinal-specific metrics
#         ordinal_metrics = {}
#         if is_ordinal:
#             ordinal_metrics = compute_ordinal_metrics(
#                 y_true[test_idx].cpu().numpy(),
#                 y_pred[test_idx].cpu().numpy()
#             )
#             print("\n--- Ordinal Metrics ---")
#             print(f"MAE: {ordinal_metrics['mae']:.4f}")
#             print(f"RMSE: {ordinal_metrics['rmse']:.4f}")
#             print(f"Spearman: {ordinal_metrics['spearman']:.4f}")
#             print(f"QWK: {ordinal_metrics['qwk']:.4f}")
#             print(f"Acc@1: {100*ordinal_metrics['acc1']:.2f}%")
#             print(f"Acc@2: {100*ordinal_metrics['acc2']:.2f}%")

#         # Save results
#         run_results = {
#             'run': run,
#             'best_epoch': best_epoch,
#             'best_val': best_val,
#             'best_test': best_test,
#             'test_accuracy': test_acc,
#             'balanced_accuracy': balanced_acc,
#             'roc_auc': roc_auc,
#         }
        
#         if ordinal_metrics:
#             run_results.update({
#                 'ordinal_mae': ordinal_metrics['mae'],
#                 'ordinal_rmse': ordinal_metrics['rmse'],
#                 'ordinal_spearman': ordinal_metrics['spearman'],
#                 'ordinal_qwk': ordinal_metrics['qwk'],
#                 'ordinal_acc1': ordinal_metrics['acc1'],
#                 'ordinal_acc2': ordinal_metrics['acc2'],
#             })
        
#         all_results.append(run_results)

#         # Save model and predictions
#         param_string = generate_param_string(args)
#         out_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/run_{run}'
#         save_results(model, y_true[test_idx].cpu().numpy(), 
#                     out[test_idx], y_true[test_idx], args, out_dir)
        
#         with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
#             json.dump(run_results, f, indent=2)

#     # Aggregate results
#     print(f"\n{'='*80}")
#     print("FINAL RESULTS (mean ± std)")
#     print(f"{'='*80}")
    
#     metrics_to_aggregate = ['best_val', 'best_test', 'test_accuracy', 
#                            'balanced_accuracy', 'roc_auc']
#     if is_ordinal:
#         metrics_to_aggregate.extend(['ordinal_mae', 'ordinal_rmse', 
#                                      'ordinal_spearman', 'ordinal_qwk', 
#                                      'ordinal_acc1', 'ordinal_acc2'])
    
#     summary = {}
#     for metric in metrics_to_aggregate:
#         values = [r[metric] for r in all_results if metric in r]
#         if values:
#             mean_val = np.mean(values)
#             std_val = np.std(values)
#             summary[f'{metric}_mean'] = float(mean_val)
#             summary[f'{metric}_std'] = float(std_val)
            
#             # Pretty print
#             if 'acc' in metric.lower():
#                 print(f"{metric}: {100*mean_val:.2f} ± {100*std_val:.2f}%")
#             else:
#                 print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

#     # Save summary
#     param_string = generate_param_string(args)
#     summary_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/summary'
#     os.makedirs(summary_dir, exist_ok=True)
    
#     with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     with open(os.path.join(summary_dir, 'all_runs.json'), 'w') as f:
#         json.dump(all_results, f, indent=2)

#     print(f"\nResults saved to: {summary_dir}")

# # =====================================================================================
# #  Section 6: Script Entry Point
# # =====================================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Fixed GNN Training Pipeline with Ordinal Losses')
#     parser_add_main_args(parser)

#     # Ordinal-specific arguments
#     parser.add_argument('--ordinal_loss', type=str, default='coral', 
#                         choices=['coral', 'dwce', 'emd', 'ce'],
#                         help="Ordinal loss: coral/dwce/emd/ce (ce=standard cross-entropy for debugging)")
#     parser.add_argument('--dwce_alpha', type=float, default=0.1,
#                         help="Alpha for DistanceWeightedCrossEntropy (default: 0.5, lower = less penalty)")
#     parser.add_argument('--early_stopping', type=int, default=0,
#                         help="Early stopping patience (0 = disabled)")

#     args = parser.parse_args()
    
#     # Print configuration
#     print("\n" + "="*80)
#     print("CONFIGURATION")
#     print("="*80)
#     print(f"Dataset: {args.dataset}")
#     print(f"Model: {args.gnn.upper()}")
#     print(f"Ordinal Loss: {args.ordinal_loss}")
#     if args.ordinal_loss == 'dwce':
#         print(f"  - Alpha: {args.dwce_alpha}")
#     print(f"Learning Rate: {args.lr}")
#     print(f"Weight Decay: {args.weight_decay}")
#     print(f"Hidden Channels: {args.hidden_channels}")
#     print(f"Dropout: {args.dropout}")
#     print(f"Epochs: {args.epochs}")
#     print(f"Runs: {args.runs}")
#     print(f"Seed: {args.seed}")
#     print("="*80 + "\n")
    
#     train_and_evaluate(args)



# # # =====================================================================================
# # #  Integrated GNN Training Pipeline (patched for ordinal evaluation + ordinal losses)
# # # =====================================================================================

# # import argparse
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, to_scipy_sparse_matrix
# # from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, balanced_accuracy_score, f1_score
# # from sklearn.metrics import cohen_kappa_score
# # from scipy.stats import spearmanr
# # from sklearn.manifold import TSNE
# # import matplotlib.pyplot as plt
# # import os
# # import sys
# # import json
# # import scipy.sparse as sp
# # import networkx as nx

# # # --- Model Imports ---
# # from model import MPNNs
# # # from FSGCN_models import FSGNN
# # # from glognn_models import MLP_NORM
# # # from gprgnn_models import GPRGNN
# # from mlp_model import MLP
# # # from xgboost_model import xgboost

# # # --- Helper Imports ---
# # from dataset import load_dataset
# # from data_utils import eval_acc, eval_rocauc, eval_prauc, class_rand_splits, load_fixed_splits, eval_balanced_acc
# # from eval import evaluate
# # from logger import Logger
# # from parse import parser_add_main_args

# # # =====================================================================================
# # #  Section 0: Which datasets are ordinal?
# # # =====================================================================================
# # ORDINAL_DATASETS = ['amazon-ratings', 'squirrel']

# # # =====================================================================================
# # #  Section 1: Helper Functions
# # # =====================================================================================

# # def fix_seed(seed=42):
# #     """Sets the seed for reproducibility."""
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # def normalize_tensor_sparse(mx, symmetric=0):
# #     """Row-normalize or symmetrically-normalize a sparse matrix."""
# #     rowsum = np.array(mx.sum(1)) + 1e-12
# #     if symmetric == 0:
# #         r_inv = np.power(rowsum, -1).flatten()
# #         r_inv[np.isinf(r_inv)] = 0.
# #         r_mat_inv = sp.diags(r_inv, 0)
# #         return r_mat_inv.dot(mx)
# #     else:
# #         r_inv = np.power(rowsum, -0.5).flatten()
# #         r_inv[np.isinf(r_inv)] = 0.
# #         r_mat_inv = sp.diags(r_inv, 0)
# #         mx = r_mat_inv.dot(mx)
# #         return mx.dot(r_mat_inv)

# # def sparse_mx_to_torch_sparse_tensor(sparse_mx):
# #     """Convert a scipy sparse matrix to a torch sparse tensor."""
# #     sparse_mx = sparse_mx.tocoo().astype(np.float32)
# #     indices = torch.from_numpy(
# #         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
# #     values = torch.from_numpy(sparse_mx.data)
# #     shape = torch.Size(sparse_mx.shape)
# #     return torch.sparse.FloatTensor(indices, values, shape)

# # def save_results(model, tsne_labels, logits, labels, args, out_dir):
# #     """Saves all run artifacts to a directory."""
# #     os.makedirs(out_dir, exist_ok=True)
# #     torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
# #     np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
# #     np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
# #     np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

# # def plot_tsne(tsne_embeds, tsne_labels, out_dir):
# #     """Generates and saves a t-SNE plot."""
# #     plt.figure(figsize=(8, 6))
# #     for c in np.unique(tsne_labels):
# #         idx = tsne_labels == c
# #         plt.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=f'Class {c}', alpha=0.6)
# #     plt.legend()
# #     plt.title('t-SNE of Node Embeddings')
# #     plt.savefig(os.path.join(out_dir, 'tsne_plot.png'))
# #     plt.close()

# # def plot_logits_distribution(logits, labels, out_dir):
# #     """Generates and saves a plot of the logits distribution."""
# #     logits = logits.cpu().detach().numpy()
# #     labels = labels.cpu().numpy().squeeze()
# #     if logits.ndim == 1:
# #         logits = logits[:, None]
# #     num_classes = logits.shape[1]
# #     plt.figure(figsize=(8, 6))
# #     for c in range(num_classes):
# #         if np.sum(labels == c) == 0:
# #             continue
# #         plt.hist(logits[labels == c, c], bins=30, alpha=0.5, label=f'Class {c}')
# #     plt.xlabel('Logit Value')
# #     plt.ylabel('Frequency')
# #     plt.title('Logits Distribution per Class')
# #     plt.legend()
# #     plt.savefig(os.path.join(out_dir, 'logits_distribution.png'))
# #     plt.close()

# # def generate_param_string(args):
# #     """
# #     Generates a unique, readable string from all relevant hyperparameters.
# #     """
# #     params = []
# #     params.append(f"metric-{args.metric}")
# #     params.append(f"lr-{args.lr}")
# #     params.append(f"wd-{args.weight_decay}")
# #     params.append(f"do-{args.dropout}")
# #     params.append(f"hid-{args.hidden_channels}")
# #     gnn_type = args.gnn
# #     if gnn_type in ['gcn', 'gat', 'sage', 'gin']:
# #         params.append(f"layers-{args.local_layers}")
# #         if gnn_type == 'gat':
# #             params.append(f"heads-{args.num_heads}")
# #         if args.res: params.append('res')
# #         if args.ln: params.append('ln')
# #         if args.bn: params.append('bn')
# #         if args.jk: params.append('jk')
# #     elif gnn_type == 'fsgcn':
# #         params.append(f"layers-{args.fsgcn_num_layers}")
# #         params.append(f"type-{args.fsgcn_feat_type}")
# #         if args.fsgcn_layer_norm: params.append('ln')
# #     elif gnn_type == 'glognn':
# #         params.append(f"alpha-{args.glognn_alpha}")
# #         params.append(f"beta-{args.glognn_beta}")
# #         params.append(f"gamma-{args.glognn_gamma}")
# #         params.append(f"delta-{args.glognn_delta}")
# #     elif gnn_type == 'gprgnn':
# #         params.append(f"K-{args.gprgnn_K}")
# #         params.append(f"alpha-{args.gprgnn_alpha}")
# #     # params.append(f"loss-{args.loss_fn}")
# #     if getattr(args, 'ordinal_loss', None):
# #         params.append(f"ordinal-{args.ordinal_loss}")
# #     return "_".join(params)

# # # =====================================================================================
# # #  Section 1.5: Ordinal Loss Implementations (work with standard logits N x C)
# # # =====================================================================================

# # class DistanceWeightedCrossEntropy(nn.Module):
# #     """
# #     Distance-weighted cross-entropy.
# #     For each true class i and predicted class j, the CE term is multiplied by weight(|i-j|).
# #     weight_fn default: 1 + alpha * distance
# #     """
# #     def __init__(self, num_classes, alpha=1.0, reduction='mean'):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         self.alpha = alpha
# #         self.reduction = reduction
# #         # Precompute distance weights matrix shape (num_classes, num_classes)
# #         dist = np.abs(np.arange(num_classes)[:, None] - np.arange(num_classes)[None, :]).astype(np.float32)
# #         self.register_buffer('weight_mat', torch.from_numpy(1.0 + self.alpha * dist))  # larger distance -> larger weight

# #     # def forward(self, logits, targets):
# #     #     """
# #     #     logits: [N, C] raw
# #     #     targets: [N] ints 0..C-1
# #     #     """
# #     #     logp = F.log_softmax(logits, dim=1)  # [N, C]
# #     #     n = logits.size(0)
# #     #     # gather per-sample weight vector for true class
# #     #     # weight_mat[true] -> [N, C]
# #     #     w = self.weight_mat[targets]  # [N, C]
# #     #     loss_per_sample = - (w * logp).sum(dim=1)  # sum_j w_ij * log p_j
# #     #     if self.reduction == 'mean':
# #     #         return loss_per_sample.mean()
# #     #     elif self.reduction == 'sum':
# #     #         return loss_per_sample.sum()
# #     #     else:
# #     #         return loss_per_sample  # none


# #     def forward(self, logits, targets):
# #         """
# #         logits: [N, C] raw
# #         targets: [N] ints 0..C-1
# #         """
# #         # Ensure weight_mat is on same device as targets
# #         wmat = self.weight_mat.to(targets.device)
    
# #         logp = F.log_softmax(logits, dim=1)  # [N, C]
    
# #         # gather per-sample weight vector
# #         w = wmat[targets]  # [N, C]
    
# #         loss_per_sample = - (w * logp).sum(dim=1)
    
# #         if self.reduction == 'mean':
# #             return loss_per_sample.mean()
# #         elif self.reduction == 'sum':
# #             return loss_per_sample.sum()
# #         else:
# #             return loss_per_sample


# # class EarthMoverDistanceLoss(nn.Module):
# #     """
# #     EMD-like loss using cumulative distribution difference (L2).
# #     preds -> softmax -> cum_pred
# #     target -> one-hot -> cum_target
# #     Loss = mean over samples of ||cum_pred - cum_target||^2
# #     """
# #     def __init__(self, reduction='mean'):
# #         super().__init__()
# #         self.reduction = reduction

# #     def forward(self, logits, targets):
# #         probs = F.softmax(logits, dim=1)  # [N, C]
# #         cum_probs = torch.cumsum(probs, dim=1)  # [N, C]
# #         # build cum_target: for class k, cum_target[k] = 1 if true_class <= k else 0
# #         # equivalently, for true class t, cum_target = [0...0, 1, 1, ...] with 1 starting at index t
# #         N, C = probs.shape
# #         device = logits.device
# #         arange = torch.arange(C, device=device).unsqueeze(0).expand(N, -1)  # [N,C]
# #         targets_exp = targets.unsqueeze(1).expand(-1, C)
# #         cum_target = (arange >= targets_exp).float()
# #         diff = cum_probs - cum_target
# #         per_sample = (diff ** 2).sum(dim=1)  # L2 over thresholds
# #         if self.reduction == 'mean':
# #             return per_sample.mean()
# #         elif self.reduction == 'sum':
# #             return per_sample.sum()
# #         else:
# #             return per_sample

# # class CORALSimplifiedLoss(nn.Module):
# #     """
# #     Simplified CORAL that uses the standard logits (N x C) without changing model.
# #     We compute probs = softmax(logits) and cumulative probabilities P_k = sum_{j<=k} p_j.
# #     For each threshold k (0..C-2), target is 1 if true_class > k else 0.
# #     Then apply BCE between P_k and target_k, summed over thresholds.
# #     """
# #     def __init__(self, reduction='mean'):
# #         super().__init__()
# #         self.reduction = reduction
# #         self.bce = nn.BCELoss(reduction='none')

# #     def forward(self, logits, targets):
# #         probs = F.softmax(logits, dim=1)  # [N, C]
# #         cum_probs = torch.cumsum(probs, dim=1)  # [N, C]
# #         # drop last column because thresholds are C-1
# #         cum_probs_thresh = cum_probs[:, :-1]  # [N, C-1]
# #         N, C = probs.shape
# #         device = logits.device
# #         arange = torch.arange(C-1, device=device).unsqueeze(0).expand(N, -1)  # [N, C-1]
# #         targets_exp = targets.unsqueeze(1).expand(-1, C-1)
# #         target_thresh = (targets_exp > arange).float()  # 1 if true_class > k
# #         # BCE between cum_probs_thresh and target_thresh
# #         loss_mat = self.bce(cum_probs_thresh, target_thresh)  # [N, C-1]
# #         per_sample = loss_mat.sum(dim=1)  # sum over thresholds
# #         if self.reduction == 'mean':
# #             return per_sample.mean()
# #         elif self.reduction == 'sum':
# #             return per_sample.sum()
# #         else:
# #             return per_sample

# # # =====================================================================================
# # #  Section 2: Ordinal metrics
# # # =====================================================================================

# # def compute_ordinal_metrics(y_true_tensor, y_pred_tensor):
# #     """Return MAE, RMSE, Spearman, QWK, Acc@1, Acc@2 (we will return a subset as requested)."""
# #     y_true = y_true_tensor.cpu().astype(int) if isinstance(y_true_tensor, np.ndarray) else y_true_tensor.cpu().numpy().astype(int)
# #     y_pred = y_pred_tensor.cpu().astype(int) if isinstance(y_pred_tensor, np.ndarray) else y_pred_tensor.cpu().numpy().astype(int)

# #     mae = np.mean(np.abs(y_true - y_pred))
# #     rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
# #     # Spearman
# #     try:
# #         spearman_corr = spearmanr(y_true, y_pred).correlation
# #         if spearman_corr is None:
# #             spearman_corr = 0.0
# #     except Exception:
# #         spearman_corr = 0.0
# #     # QWK
# #     try:
# #         qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
# #     except Exception:
# #         qwk = 0.0
# #     acc1 = np.mean(np.abs(y_true - y_pred) <= 1)
# #     acc2 = np.mean(np.abs(y_true - y_pred) <= 2)
# #     return {
# #         "mae": float(mae),
# #         "rmse": float(rmse),
# #         "spearman": float(spearman_corr),
# #         "qwk": float(qwk),
# #         "acc1": float(acc1),
# #         "acc2": float(acc2)
# #     }

# # # =====================================================================================
# # #  Section 3: Main Training and Evaluation Function
# # # =====================================================================================

# # def train_and_evaluate(args):
# #     """Main function to run the training and evaluation pipeline."""
# #     fix_seed(args.seed)
# #     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
# #     print('device', device)

# #     dataset = load_dataset(args.data_dir, args.dataset)
# #     if len(dataset.label.shape) == 1:
# #         dataset.label = dataset.label.unsqueeze(1)

# #     if args.rand_split:
# #         split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]
# #     else:
# #         split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

# #     dataset.label = dataset.label.to(device)

# #     n, c, d = dataset.graph['num_nodes'], max(dataset.label.max().item() + 1, dataset.label.shape[1]), dataset.graph['node_feat'].shape[1]
# #     print(f'Dataset: {args.dataset}, Nodes: {n}, Classes: {c}, Features: {d}')

# #     # --- Pre-computation and Model Initialization ---
# #     fsgcn_list_mat = None
# #     glognn_adj_sparse, glognn_adj_dense, glognn_features = None, None, None
# #     gprgnn_features = None

# #     # handle special GNN branches (unchanged)
# #     if args.gnn == 'fsgcn':
# #         # (kept identical to your original)
# #         print("--- FSGCN: Starting feature pre-computation ---")
# #         edge_index_no_loops, _ = remove_self_loops(dataset.graph['edge_index'])
# #         adj = to_scipy_sparse_matrix(edge_index_no_loops, num_nodes=n)
# #         adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

# #         edge_index_with_loops, _ = add_self_loops(edge_index_no_loops, num_nodes=n)
# #         adj_i = to_scipy_sparse_matrix(edge_index_with_loops, num_nodes=n)
# #         adj_i = sparse_mx_to_torch_sparse_tensor(adj_i).to(device)

# #         list_mat = [dataset.graph['node_feat'].to(device)]
# #         no_loop_mat = loop_mat = dataset.graph['node_feat'].to(device)

# #         for _ in range(args.fsgcn_num_layers):
# #             no_loop_mat = torch.spmm(adj, no_loop_mat)
# #             loop_mat = torch.spmm(adj_i, loop_mat)
# #             list_mat.append(no_loop_mat)
# #             list_mat.append(loop_mat)

# #         if args.fsgcn_feat_type == "homophily":
# #             select_idx = [0] + [2 * ll for ll in range(1, args.fsgcn_num_layers + 1)]
# #             fsgcn_list_mat = [list_mat[ll] for ll in select_idx]
# #         elif args.fsgcn_feat_type == "heterophily":
# #             select_idx = [0] + [2 * ll - 1 for ll in range(1, args.fsgcn_num_layers + 1)]
# #             fsgcn_list_mat = [list_mat[ll] for ll in select_idx]
# #         else:
# #             fsgcn_list_mat = list_mat

# #         # instantiate your FSGNN model (commented out earlier) - keep as is if available
# #         # model = FSGNN(...)
# #         # For safety, if not available, fallback to MPNNs (user must ensure FSGNN exists)
# #         model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
# #                       heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
# #                       res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn='gcn').to(device)
# #         print(f"--- FSGCN: Pre-computation finished. Using {len(fsgcn_list_mat)} feature matrices. ---")

# #     elif args.gnn == 'glognn':
# #         print("--- GloGNN: Pre-computation started ---")
# #         glognn_features = dataset.graph['node_feat'].to(device)
# #         edge_index_np = dataset.graph['edge_index'].cpu().numpy().T
# #         adj = nx.adjacency_matrix(nx.from_edgelist(edge_index_np), nodelist=range(n))
# #         glognn_adj_sparse = sparse_mx_to_torch_sparse_tensor(adj).to(device)
# #         glognn_adj_dense = glognn_adj_sparse.to_dense().to(device)
# #         # model = MLP_NORM(...)  # keep original implementation if present
# #         model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
# #                       heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
# #                       res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn='gcn').to(device)
# #         print("--- GloGNN: Pre-computation finished ---")

# #     elif args.gnn == 'gprgnn':
# #         print("--- GPRGNN: Pre-computation started ---")
# #         features_np = dataset.graph['node_feat'].cpu().numpy()
# #         normalized_features = normalize_tensor_sparse(sp.csr_matrix(features_np), symmetric=0).todense()
# #         gprgnn_features = torch.FloatTensor(normalized_features).to(device)
# #         dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index']).to(device)
# #         # model = GPRGNN(...)  # keep original if present
# #         model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
# #                       heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
# #                       res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn='gcn').to(device)
# #         print("--- GPRGNN: Pre-computation finished ---")

# #     elif args.gnn == 'mlp':
# #         print("--- MLP: Initializing Model ---")
# #         dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
# #         model = MLP(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout).to(device)

# #     else:
# #         dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
# #         dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
# #         dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
# #         dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
# #         dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
# #         model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
# #                       heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
# #                       res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn=args.gnn).to(device)

# #     # --- Loss/criterion selection (including ordinal losses)
# #     is_ordinal = args.dataset in ORDINAL_DATASETS
# #     if is_ordinal:
# #         # We'll create a loss wrapper based on args.ordinal_loss
# #         if args.ordinal_loss == 'dwce':
# #             criterion = DistanceWeightedCrossEntropy(num_classes=c, alpha=args.dwce_alpha)
# #             print(f"Using DistanceWeightedCrossEntropy (alpha={args.dwce_alpha}) for ordinal dataset {args.dataset}")
# #         elif args.ordinal_loss == 'emd':
# #             criterion = EarthMoverDistanceLoss()
# #             print(f"Using EarthMoverDistanceLoss for ordinal dataset {args.dataset}")
# #         else:  # 'coral' simplified (default)
# #             criterion = CORALSimplifiedLoss()
# #             print(f"Using CORAL-simplified loss for ordinal dataset {args.dataset}")
# #     else:
# #         # non-ordinal: keep your original defaults
# #         if args.dataset in ('questions'):
# #             criterion = nn.BCEWithLogitsLoss()
# #         else:
# #             # multi-class classification
# #             criterion = nn.NLLLoss()

# #     eval_func = {'prauc': eval_prauc, 'rocauc': eval_rocauc, 'balacc': eval_balanced_acc}.get(args.metric, eval_acc)
# #     logger = Logger(args.runs, args)

# #     all_acc, vals, all_balanced_acc, all_roc_auc, all_pr_auc = [], [], [], [], []
# #     all_reports_default, all_reports_optimal = [], []

# #     for run in range(args.runs):
# #         split_idx = split_idx_lst[run]
# #         train_idx = split_idx['train'].to(device)
# #         model.reset_parameters()

# #         # --- Conditional Optimizer Setup ---
# #         if args.gnn == 'gprgnn':
# #             optimizer = torch.optim.Adam([
# #                 {'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
# #                 {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
# #                 {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.lr}
# #             ], lr=args.lr)
# #         else:
# #             optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

# #         best_val, best_test = float('-inf'), float('-inf')
# #         best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

# #         for epoch in range(args.epochs):
# #             model.train()
# #             optimizer.zero_grad()

# #             # --- Conditional Forward Pass ---
# #             if args.gnn == 'fsgcn':
# #                 out = model(fsgcn_list_mat)
# #             elif args.gnn == 'glognn':
# #                 out = model(glognn_features, glognn_adj_sparse, glognn_adj_dense)
# #             elif args.gnn == 'gprgnn':
# #                 out = model(gprgnn_features, dataset.graph['edge_index'])
# #             elif args.gnn == 'mlp':
# #                 out = model(dataset.graph['node_feat'])
# #             else:
# #                 out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

# #             # NOTE: 'out' shape is expected [N, C] logits for classification/ordinal losses (we did not change models)
# #             if is_ordinal:
# #                 # targets are integers 0..C-1
# #                 targets_train = dataset.label.squeeze(1)[train_idx].long()
# #                 loss = criterion(out[train_idx], targets_train)
# #             else:
# #                 if args.dataset in ('questions'):
# #                     true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1) if dataset.label.shape[1] == 1 else dataset.label
# #                     loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
# #                 else:
# #                     out_log_softmax = F.log_softmax(out, dim=1)
# #                     loss = criterion(out_log_softmax[train_idx], dataset.label.squeeze(1)[train_idx])

# #             loss.backward()
# #             optimizer.step()

# #             with torch.no_grad():
# #                 # Use your existing evaluate() function which returns (train, val, test, loss)
# #                 result = evaluate(model, dataset, split_idx, eval_func, criterion, args,
# #                                   fsgcn_list_mat=fsgcn_list_mat,
# #                                   glognn_features=glognn_features, glognn_adj_sparse=glognn_adj_sparse, glognn_adj_dense=glognn_adj_dense,
# #                                   gprgnn_features=gprgnn_features)

# #             # result expected: (train_metric, val_metric, test_metric, train_loss)
# #             logger.add_result(run, result[:-1])

# #             if result[1] > best_val:
# #                 best_val, best_test = result[1], result[2]
# #                 best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

# #             if epoch % args.display_step == 0:
# #                 print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100*result[0]:.2f}%, Valid: {100*result[1]:.2f}%, Test: {100*result[2]:.2f}%')

# #         print(f'Run {run+1}/{args.runs}: Best Valid: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')

# #         # --- Final Evaluation ---
# #         model.load_state_dict(best_model_state)
# #         model.eval()
# #         with torch.no_grad():
# #             if args.gnn == 'fsgcn':
# #                 out = model(fsgcn_list_mat)
# #             elif args.gnn == 'glognn':
# #                 out = model(glognn_features, glognn_adj_sparse, glognn_adj_dense)
# #             elif args.gnn == 'gprgnn':
# #                 out = model(gprgnn_features, dataset.graph['edge_index'])
# #             elif args.gnn == 'mlp':
# #                 out = model(dataset.graph['node_feat'])
# #             else:
# #                 out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

# #         y_true, test_idx = dataset.label, split_idx['test']
# #         best_threshold, report_optimal = 0.5, None

# #         # Compute probs and predicted classes for downstream metrics
# #         out_log_softmax = F.log_softmax(out, dim=1)
# #         pred_probs = torch.exp(out_log_softmax)  # [N, C]
# #         y_pred_default = pred_probs.argmax(dim=-1)  # int predictions

# #         # Standard metrics (kept)
# #         if c > 2:
# #             probs = torch.exp(F.log_softmax(out[test_idx], dim=1))
# #             y_pred = probs.argmax(dim=-1)
# #             roc_auc = roc_auc_score(y_true[test_idx].cpu().numpy(), probs.detach().cpu().numpy(), multi_class='ovr')
# #             pr_auc = eval_prauc(y_true[test_idx], F.log_softmax(out[test_idx], dim=1))
# #             balanced_acc = balanced_accuracy_score(y_true[test_idx].cpu().numpy(), y_pred_default[test_idx].cpu().numpy())
# #             report_default = classification_report(y_true[test_idx].cpu().numpy(), y_pred_default[test_idx].cpu().numpy(), output_dict=True)
# #         else:
# #             roc_auc = eval_rocauc(y_true[test_idx], out_log_softmax[test_idx])
# #             probs_test = torch.exp(out_log_softmax[test_idx])
# #             pr_auc = average_precision_score(y_true[test_idx].cpu().numpy(), probs_test[:,1].detach().cpu().numpy())

# #             valid_idx = split_idx['valid']
# #             valid_probs = torch.exp(out_log_softmax[valid_idx])[:, 1].cpu().numpy()
# #             valid_true = y_true[valid_idx].cpu().numpy()
# #             best_f1 = -1
# #             for threshold in np.linspace(0, 1, 100):
# #                 f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
# #                 if f1 > best_f1:
# #                     best_f1, best_threshold = f1, threshold
# #             print(f'Optimal Threshold on Val Set: {best_threshold:.4f} (Macro F1: {best_f1:.4f})')

# #             y_true_test_np = y_true[test_idx].cpu().numpy()
# #             y_pred_default_np = y_pred_default[test_idx].cpu().numpy()
# #             balanced_acc = balanced_accuracy_score(y_true_test_np, y_pred_default_np)
# #             report_default = classification_report(y_true_test_np, y_pred_default_np, output_dict=True)
# #             report_optimal = None

# #         # Ordinal metrics (only for ordinal datasets)
# #         ordinal_metrics = {}
# #         if is_ordinal:
# #             # y_true integers and y_pred_default are already int tensors
# #             y_true_test = y_true[test_idx].squeeze(1)
# #             y_pred_test = y_pred_default[test_idx]
# #             ordinal_metrics = compute_ordinal_metrics(y_true_test, y_pred_test)
# #             # print the selected ordinal metrics (you asked for MAE, Spearman, QWK)
# #             print("\n--- Ordinal Metrics (selected) ---")
# #             print(f"MAE: {ordinal_metrics['mae']:.4f}")
# #             print(f"Spearman: {ordinal_metrics['spearman']:.4f}")
# #             print(f"QWK: {ordinal_metrics['qwk']:.4f}")

# #         print(f'AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}, Balanced Acc: {balanced_acc:.4f}')
# #         print('\n--- Classification Report (Default Threshold) ---\n', json.dumps(report_default, indent=2))
# #         if report_optimal: print('\n--- Classification Report (Optimal Threshold) ---\n', json.dumps(report_optimal, indent=2))

# #         all_acc.append(best_test)
# #         vals.append(best_val)
# #         all_balanced_acc.append(balanced_acc)
# #         all_roc_auc.append(roc_auc)
# #         all_pr_auc.append(pr_auc)
# #         all_reports_default.append(report_default)
# #         if report_optimal: all_reports_optimal.append(report_optimal)

# #         # --- Save results (minimal)
# #         with torch.no_grad():
# #             labels_cpu = y_true[test_idx].cpu().numpy().squeeze()

# #         param_string = generate_param_string(args)
# #         out_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/run_{run}'
# #         print('save results started')
# #         save_results(model, labels_cpu, out[split_idx['test']], y_true[split_idx['test']], args, out_dir)
# #         print('save results finished')

# #         metrics = {
# #             'accuracy_vals': best_val,
# #             'accuracy_from_val': best_test,
# #             'balanced_accuracy': balanced_acc,
# #             'auc_roc': roc_auc,
# #             'auc_pr': pr_auc,
# #             'optimal_threshold': best_threshold if c <= 2 else None,
# #             'classification_report_default': report_default,
# #             'classification_report_optimal': report_optimal,
# #         }
# #         # add ordinal metrics if present
# #         if ordinal_metrics:
# #             metrics.update({
# #                 'ordinal_mae': ordinal_metrics['mae'],
# #                 'ordinal_spearman': ordinal_metrics['spearman'],
# #                 'ordinal_qwk': ordinal_metrics['qwk'],
# #                 'ordinal_acc1': ordinal_metrics['acc1'],
# #                 'ordinal_acc2': ordinal_metrics['acc2'],
# #             })

# #         with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
# #             json.dump(metrics, f, indent=2)

# #     # --- Final Summary ---
# #     summary = {
# #         'accuracy_mean_vals': float(np.mean(vals)), 'accuracy_std_vals': float(np.std(vals)),
# #         'accuracy_mean': float(np.mean(all_acc)), 'accuracy_std': float(np.std(all_acc)),
# #         'balanced_accuracy_mean': float(np.mean(all_balanced_acc)), 'balanced_accuracy_std': float(np.std(all_balanced_acc)),
# #         'auc_roc_mean': float(np.mean(all_roc_auc)), 'auc_roc_std': float(np.std(all_roc_auc)),
# #         'auc_pr_mean': float(np.mean(all_pr_auc)), 'auc_pr_std': float(np.std(all_pr_auc)),
# #     }

# #     param_string = generate_param_string(args)
# #     summary_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/summary'
# #     os.makedirs(summary_dir, exist_ok=True)
# #     with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
# #         json.dump(summary, f, indent=2)
# #     with open(os.path.join(summary_dir, 'reports_default.json'), 'w') as f:
# #         json.dump(all_reports_default, f, indent=2)
# #     if all_reports_optimal:
# #         with open(os.path.join(summary_dir, 'reports_optimal.json'), 'w') as f:
# #             json.dump(all_reports_optimal, f, indent=2)

# # # =====================================================================================
# # #  Section 4: Script Entry Point
# # # =====================================================================================

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description='Integrated GNN Training Pipeline')
# #     parser_add_main_args(parser)

# #     # Ordinal-specific flags
# #     parser.add_argument('--ordinal_loss', type=str, default='coral', choices=['coral', 'dwce', 'emd'],
# #                         help="Ordinal loss to use for ordinal datasets (coral/dwce/emd). coral is simplified CORAL that works with standard logits.")
# #     parser.add_argument('--dwce_alpha', type=float, default=1.0,
# #                         help="Alpha used in DistanceWeightedCrossEntropy weight = 1 + alpha * distance (only for 'dwce').")

# #     args = parser.parse_args()
# #     print(args)
# #     train_and_evaluate(args)
