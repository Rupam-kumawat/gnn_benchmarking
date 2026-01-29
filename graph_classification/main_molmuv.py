
#!/usr/bin/env python3
"""
Unified Pipeline for Graph Classification
Extended to fully support OGBG-MolMUV (multi-task binary with missing labels)
Integrated with GPS and Subgraphormer.

Usage example:
python main_graph_gnn_weighted_molmuv.py --dataset ogbg-molmuv --dataset_type ogb \
    --model_name gps --model_family gt --epochs 100 --num_layers 4 --hidden_channels 64 \
    --dropout 0.0 --pool mean --metric rocauc --batch_size 32 --device 0 --runs 3 \
    --use_class_weight --pos_weight_cap 1000.0 --use_lap_pe --lap_pe_dim 16
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from parse_graphclass import parser_add_main_args
from gnn_models import GCN, GAT, SAGE, GIN
# --- Added Imports for GPS and Subgraphormer ---
# from gps_model import GPSModel
from subgraphormer_model import Subgraphormer
from subgraphormer_utils import get_subgraphormer_transform
from pe_layer import compute_laplacian_pe

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# ===================== Loss functions =====================
class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for logits input (compatible with BCEWithLogitsLoss semantics).
    Supports arbitrary shape (e.g., [B, num_tasks]).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Initialize with reduction='none' to get per-element losses
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # logits: float tensor, targets: float tensor (0/1)
        loss_bce = self.bce_loss(logits, targets)
        pt = torch.exp(-loss_bce)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * loss_bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # Return unreduced loss
            return loss


class FocalLoss(nn.Module):
    """
    Multiclass focal loss expecting log-probabilities (i.e., input should be log_softmax).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets):
        # log_probs: [N, C] log-softmax; targets: long [N]
        ce = F.nll_loss(log_probs, targets, reduction='none')  # negative log-prob
        pt = torch.exp(-ce)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = ce * focal_term
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha).to(log_probs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ===================== Utilities =====================

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_param_string(args):
    params = [f"lr-{args.lr}", f"pool-{args.pool}", f"metric-{args.metric}", f"hid-{args.hidden_channels}", f"do-{args.dropout}",
              f"layers-{args.num_layers}"]
    if args.model_family == 'gt' or args.model_name == 'gat' or args.model_name == 'subgraphormer':
        params.append(f"heads-{args.nhead}")
    if args.model_name == 'gps':
         params.append(f"local-{args.local_gnn}")
         if args.use_lap_pe: params.append(f"lapPE-{args.lap_pe_dim}")
    if args.model_family == 'gnn':
        params.append(f"bn-{args.use_bn}")
        params.append(f"res-{args.use_residual}")
    if args.use_class_weight: params.append('class_weight')
    if args.use_focal_loss: params.append(f'focal_g-{args.focal_gamma}')
    if args.use_focal_loss and args.dataset_type == 'ogb':
        params.append(f'focal_a-{args.focal_alpha}')
    return "_".join(params)


# ===================== Training and Evaluation =====================

def train(model, loader, optimizer, device, criterion, dataset_type, num_tasks, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # shape [B, num_tasks] or [B, C]

        # ==================== MODIFIED SECTION START ====================
        if dataset_type == 'ogb':
            y_true = data.y.to(torch.float)
            mask = (y_true != -1)
            
            if mask.sum() == 0:
                # nothing to learn on this batch
                continue
            
            # UNIFIED LOSS CALCULATION
            loss_unreduced = criterion(out, y_true)
            
            # Apply mask and compute mean
            loss = (loss_unreduced * mask.float()).sum() / mask.sum()
            
        else:
            # TU Dataset logic (remains unchanged)
            y_true = data.y.squeeze().to(torch.long)
            out_log_softmax = F.log_softmax(out, dim=1)
            loss = criterion(out_log_softmax, y_true)
        # ===================== MODIFIED SECTION END =====================

        loss.backward()
        # CRITICAL: Add gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

    if total_graphs == 0:
        return 0.0
    return total_loss / total_graphs


def evaluate_graphclass(model, loader, device, dataset_type, evaluator=None):
    """
    Runs evaluation and returns (y_true_tensor, y_pred_logits_tensor, metrics_dict)
    """
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            # ensure out is a 2D tensor
            if out.dim() == 1:
                out = out.unsqueeze(1)
            preds.append(out.cpu())
            ys.append(data.y.cpu())

    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(preds, dim=0)

    metrics = {}

    if dataset_type == 'ogb':
        # OGB evaluator expects numpy arrays
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        # Replace NaN with -1 if present
        try:
            ev_res = evaluator.eval({"y_true": y_true_np, "y_pred": y_pred_np})
        except Exception as e:
            # fallback: compute roc auc per task ignoring missing labels
            num_tasks = y_true_np.shape[1]
            rocs = []
            aps = []
            for t in range(num_tasks):
                mask = y_true_np[:, t] != -1
                if mask.sum() < 2:
                    rocs.append(float('nan'))
                    aps.append(float('nan'))
                    continue
                try:
                    rocs.append(roc_auc_score(y_true_np[mask, t], y_pred_np[mask, t]))
                except Exception:
                    rocs.append(float('nan'))
                try:
                    aps.append(average_precision_score(y_true_np[mask, t], y_pred_np[mask, t]))
                except Exception:
                    aps.append(float('nan'))
            ev_res = {"rocauc": np.nanmean(rocs), "ap": np.nanmean(aps)}
        # Map common keys
        metrics['rocauc'] = ev_res.get('rocauc', ev_res.get('roc_auc', None))
        metrics['ap'] = ev_res.get('ap', ev_res.get('prauc', None))
        metrics['acc'] = None
        metrics['balacc'] = None
    else:
        # TU dataset
        y_true_np = y_true.squeeze().numpy()
        if y_pred.shape[1] == 1:
            probs = torch.sigmoid(y_pred).squeeze().numpy()
            y_pred_binary = (probs >= 0.5).astype(int)
            metrics['acc'] = accuracy_score(y_true_np, y_pred_binary)
            metrics['balacc'] = balanced_accuracy_score(y_true_np, y_pred_binary)
            try:
                metrics['rocauc'] = roc_auc_score(y_true_np, probs)
            except Exception:
                metrics['rocauc'] = None
            metrics['ap'] = None
        else:
            preds_arg = torch.argmax(y_pred, dim=1).numpy()
            metrics['acc'] = accuracy_score(y_true_np, preds_arg)
            metrics['balacc'] = balanced_accuracy_score(y_true_np, preds_arg)
            metrics['rocauc'] = None
            metrics['ap'] = None

    return y_true, y_pred, metrics


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification')
    parser_add_main_args(parser)

    # Extra options for class weighting / focal loss
    parser.add_argument('--use_class_weight', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--pos_weight_cap', type=float, default=10.0, help='Maximum value for pos_weight')

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(args.seed)
    print(args)

    # --- Added Transform Logic for Subgraphormer ---
    transform = None
    if args.model_name == 'subgraphormer':
        print("Using Subgraphormer transform...")
        transform = get_subgraphormer_transform(args)

    # Data loading
    # Data loading
    if args.dataset_type == 'ogb':
        # --- FIXED SECTION START ---
        if hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals([DataEdgeAttr]):
                dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB', transform=transform)
        else:
            # Fallback for older PyTorch versions where safe_globals doesn't exist
            dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB', transform=transform)
        # --- FIXED SECTION END ---
        
        # --- Added LapPE Computation for OGB ---
        if args.use_lap_pe and args.model_name != 'subgraphormer':
             print("Pre-computing Laplacian PE for OGB dataset...")
             for g in tqdm(dataset, desc="Computing LapPE"):
                 compute_laplacian_pe(g, args.lap_pe_dim)

        split_idx = dataset.get_idx_split()

        # Convert NaN to -1 (OGB convention for missing labels)
        data_obj = dataset.data
        if hasattr(data_obj, 'y'):
            try:
                y_temp = data_obj.y
                # if float and contains nan
                if torch.is_floating_point(y_temp) and torch.isnan(y_temp).any():
                    y_temp = y_temp.clone()
                    y_temp[torch.isnan(y_temp)] = -1
                    dataset.data.y = y_temp
            except Exception:
                pass

        train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, shuffle=False)
        evaluator = Evaluator(name=args.dataset)
        num_tasks = dataset.num_tasks
        num_features = dataset.num_features
        # FIXED: Only consider binary if single task
        is_binary = (num_tasks == 1)
        print(f"Dataset: {args.dataset}, num_tasks: {num_tasks}, is_binary: {is_binary}")
    else:
        dataset = TUDataset(root='data/TUDataset', name=args.dataset, transform=transform)
        
        # --- Added LapPE Computation for TU ---
        if args.use_lap_pe and args.model_name != 'subgraphormer':
            print("Pre-computing Laplacian PE for TU dataset...")
            for g in tqdm(dataset, desc="Computing LapPE"):
                compute_laplacian_pe(g, args.lap_pe_dim)

        # Only apply OneHotDegree if features are missing AND we aren't using a custom transform (like Subgraphormer)
        if dataset.num_node_features == 0 and not transform:
            max_deg = 0
            for g in dataset:
                if g.edge_index is not None and g.edge_index.numel() > 0:
                    d = int(degree(g.edge_index[0], num_nodes=g.num_nodes).max())
                    if d > max_deg: max_deg = d
            dataset.transform = OneHotDegree(max_degree=max_deg)

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        test_size = len(dataset) // 10
        valid_size = len(dataset) // 10
        test_dataset = dataset[indices[:test_size]]
        valid_dataset = dataset[indices[test_size:test_size + valid_size]]
        train_dataset = dataset[indices[test_size + valid_size:]]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        evaluator = None
        num_tasks = dataset.num_classes
        num_features = dataset.num_features
        is_binary = dataset.num_classes == 2

    # ==================== MODIFIED SECTION START ====================
    # Loss setup
    class_weights, pos_weight = None, None
    if args.use_class_weight:
        print('--- Calculating class weights ---')
        # gather all labels
        if args.dataset_type == 'ogb':
            y_all = dataset.data.y
            y_all = y_all.float()
            # pos weight per task
            pos_samples = (y_all == 1).sum(dim=0).float()
            total = (y_all != -1).sum(dim=0).float()
            # Avoid division by zero
            pos_weight = ((total - pos_samples) / (pos_samples + 1e-8))
            # CRITICAL FIX: Cap pos_weight to prevent explosion
            pos_weight = torch.clamp(pos_weight, min=1.0, max=args.pos_weight_cap)
            pos_weight = pos_weight.to(device)
            print(f'pos_weight (capped at {args.pos_weight_cap}):', pos_weight)
        else:
            y_all = torch.cat([d.y for d in dataset], dim=0).squeeze().to(torch.long)
            class_counts = torch.bincount(y_all)
            class_weights = y_all.shape[0] / (num_tasks * class_counts.float())
            class_weights[torch.isinf(class_weights)] = 0.0
            class_weights = class_weights.to(device)
            print('class_weights:', class_weights)

    if args.dataset_type == 'ogb':
        if args.use_focal_loss:
            print(f'Using BinaryFocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma})')
            # CRITICAL: Set reduction='none' so we can apply the OGB mask manually
            criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='none')
        else:
            print('Using BCEWithLogitsLoss (with pos_weight)')
            # CRITICAL: Set reduction='none' so we can apply the OGB mask manually
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    else:
        # TUDataset logic
        if args.use_focal_loss:
            print('Using Multiclass FocalLoss')
            # Keep original 'mean' reduction for non-OGB
            criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, reduction='mean')
        else:
            print('Using NLLLoss')
            # Keep original 'mean' reduction for non-OGB
            criterion = nn.NLLLoss(weight=class_weights, reduction='mean')
    # ===================== MODIFIED SECTION END =====================

    # Tracking
    all_best_valid = []
    all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
    all_reports_default, all_reports_optimal = [], []

    for run in range(1, args.runs + 1):
        print(f"--- Run {run}/{args.runs} ---")
        models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE, 'gin': GIN}
        ModelClass = models.get(args.model_name)
        use_ogb = args.dataset_type == 'ogb'

        if args.model_family == 'gnn':
            model = ModelClass(num_features, num_tasks, hidden=args.hidden_channels, num_layers=args.num_layers,
                               dropout=args.dropout, pool=args.pool, use_ogb_features=use_ogb, use_bn=args.use_bn,
                               use_residual=args.use_residual).to(device)
        else:
            # --- Added GPS and Subgraphormer Instantiation ---
            if args.model_name == 'gps':
                model = GPSModel(
                    num_features=num_features, num_classes=num_tasks,
                    hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                    nhead=args.nhead, dropout=args.dropout, pool=args.pool,
                    local_gnn_type=args.local_gnn, global_model_type=args.global_model,
                    use_ogb_features=use_ogb, use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim
                ).to(device)
            elif args.model_name == 'subgraphormer':
                model = Subgraphormer(num_features, num_tasks, args, dataset_type=args.dataset_type).to(device)
            else:
                raise ValueError(f"Unknown model name: {args.model_name} for family {args.model_family}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        best_valid_metric = -1.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, device, criterion, args.dataset_type, num_tasks, args.grad_clip)
            yv_true, yv_pred, valid_perf = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
            print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Valid perf: {valid_perf}")
            metric_val = valid_perf.get(args.metric, -1)
            if metric_val is None:
                metric_val = -1
            if metric_val > best_valid_metric:
                best_valid_metric = metric_val
                best_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Step the scheduler
            scheduler.step(metric_val)

        print(f"Run {run} finished. Best validation {args.metric}: {best_valid_metric}")
        all_best_valid.append(best_valid_metric)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final eval
        y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
        y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

        report_optimal = None
        # FIXED: Only do threshold optimization for single-task binary classification
        if is_binary and num_tasks == 1:
            print("Performing threshold optimization for single-task binary classification...")
            # compute optimal threshold on validation
            if y_valid_pred.shape[1] == 1:
                valid_probs = torch.sigmoid(y_valid_pred).squeeze().numpy()
            else:
                valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].numpy()
            valid_true = y_valid_true.numpy().squeeze()
            # mask missing labels
            mask_valid = valid_true != -1
            if mask_valid.sum() > 0:
                best_f1 = -1.0
                best_threshold = 0.5
                for th in np.linspace(0, 1, 101):
                    f1 = f1_score(valid_true[mask_valid].astype(int), (valid_probs[mask_valid] >= th).astype(int), average='macro', zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_threshold = f1, th
                print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
                # apply to test
                if y_test_pred.shape[1] == 1:
                    test_probs_np = torch.sigmoid(y_test_pred).squeeze().numpy()
                else:
                    test_probs_np = torch.softmax(y_test_pred, dim=1)[:, 1].numpy()
                y_test_true_np = y_test_true.numpy().squeeze()
                mask_test = y_test_true_np != -1
                if mask_test.sum() > 0:
                    y_pred_opt = (test_probs_np[mask_test] >= best_threshold).astype(int)
                    y_true_opt = y_test_true_np[mask_test].astype(int)
                    test_acc_opt = accuracy_score(y_true_opt, y_pred_opt)
                    test_bal_acc_opt = balanced_accuracy_score(y_true_opt, y_pred_opt)
                    report_optimal = classification_report(y_true_opt, y_pred_opt, output_dict=True, zero_division=0)
                    test_metrics_default['acc'] = test_acc_opt
                    test_metrics_default['balacc'] = test_bal_acc_opt
                    print(f"Test Accuracy (optimal): {test_acc_opt:.4f}, Balanced Acc: {test_bal_acc_opt:.4f}")

        # aggregated metrics and saving
        acc = test_metrics_default.get('acc', None)
        bal_acc = test_metrics_default.get('balacc', None)
        roc_auc = test_metrics_default.get('rocauc', None)
        pr_auc = test_metrics_default.get('ap', test_metrics_default.get('prauc', None))

        # FIXED: Only compute classification report for single-task or non-OGB datasets
        report_default = None
        if num_tasks == 1 or args.dataset_type != 'ogb':
            try:
                if y_test_pred.shape[1] > 1:
                    y_pred_default_np = torch.argmax(y_test_pred, dim=1).numpy()
                else:
                    # binary default: threshold 0.5
                    y_pred_default_np = (torch.sigmoid(y_test_pred).squeeze().numpy() >= 0.5).astype(int)
                y_test_true_np = y_test_true.numpy()
                report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
            except Exception as e:
                print(f"Warning: Could not generate classification report: {e}")
                report_default = None
        else:
            print("Skipping classification report for multi-task binary classification (use ROC-AUC and AP instead)")

        all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
        all_reports_default.append(report_default)
        if report_optimal: all_reports_optimal.append(report_optimal)

        param_string = generate_param_string(args)
        out_dir = f'results_weighted/{args.dataset}/{args.model_name}/{param_string}/run_{run}'
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))

        run_metrics = {'best_validation_metric': best_valid_metric, 'accuracy': acc, 'balanced_accuracy': bal_acc,
                       'auc_roc': roc_auc, 'auc_pr': pr_auc,
                       'classification_report_default': report_default,
                       'classification_report_optimal': report_optimal}
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(run_metrics, f, indent=2)
        
        # print(f"Run {run} Test Metrics: ROC-AUC={roc_auc:.4f if roc_auc else 'N/A'}, AP={pr_auc:.4f if pr_auc else 'N/A'}")

    # summary
    param_string = generate_param_string(args)
    summary_dir = f'results_weighted/{args.dataset}/{args.model_name}/{param_string}/summary'
    os.makedirs(summary_dir, exist_ok=True)
    summary = {
        'metric': args.metric,
        'best_valid_mean': float(np.nanmean(all_best_valid)), 'best_valid_std': float(np.nanstd(all_best_valid)),
        'accuracy_mean': float(np.nanmean([v for v in all_acc if v is not None])) if any(v is not None for v in all_acc) else None,
        'accuracy_std': float(np.nanstd([v for v in all_acc if v is not None])) if any(v is not None for v in all_acc) else None,
        'balanced_accuracy_mean': float(np.nanmean([v for v in all_bal_acc if v is not None])) if any(v is not None for v in all_bal_acc) else None,
        'balanced_accuracy_std': float(np.nanstd([v for v in all_bal_acc if v is not None])) if any(v is not None for v in all_bal_acc) else None,
        'auc_roc_mean': float(np.nanmean([v for v in all_roc_auc if v is not None])) if any(v is not None for v in all_roc_auc) else None,
        'auc_roc_std': float(np.nanstd([v for v in all_roc_auc if v is not None])) if any(v is not None for v in all_roc_auc) else None,
        'auc_pr_mean': float(np.nanmean([v for v in all_pr_auc if v is not None])) if any(v is not None for v in all_pr_auc) else None,
        'auc_pr_std': float(np.nanstd([v for v in all_pr_auc if v is not None])) if any(v is not None for v in all_pr_auc) else None,
    }
    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(summary_dir, 'reports_default.json'), 'w') as f:
        json.dump(all_reports_default, f, indent=2)
    if all_reports_optimal:
        with open(os.path.join(summary_dir, 'reports_optimal.json'), 'w') as f:
            json.dump(all_reports_optimal, f, indent=2)

    print('\n' + '='*40 + '\nFINAL SUMMARY\n' + '='*40)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()