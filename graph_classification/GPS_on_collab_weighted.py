import argparse, importlib, os, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from ogb.graphproppred import PygGraphPropPredDataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
from torcheval.metrics import BinaryAUPRC, MulticlassAUPRC
from torch_geometric.datasets import TUDataset
import json
from collections import Counter

import random
random.seed(42) # For reproducibility

# ------------------------------
# Safe globals patch for torch.load (PyTorch >= 2.6)
# ------------------------------
if torch.__version__ >= "2.6.0":
    import torch_geometric.data
    from torch_geometric.data import Data, Batch
    safe_globals = [Data, Batch]
    optional_classes = ["DataEdgeAttr", "DataTensorAttr", "GlobalStorage"]
    for cls_name in optional_classes:
        try:
            cls = getattr(torch_geometric.data, cls_name, None)
            if cls is None:
                for submod in ["torch_geometric.data.data", "torch_geometric.data.storage"]:
                    try:
                        mod = importlib.import_module(submod)
                        cls = getattr(mod, cls_name, None)
                        if cls is not None:
                            break
                    except ImportError:
                        pass
            if cls is not None:
                safe_globals.append(cls)
        except Exception:
            pass
    torch.serialization.add_safe_globals(safe_globals)

# ------------------------------
# ADDED: Loss Function Classes
# ------------------------------

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss, compatible with `BCEWithLogitsLoss` (expects raw logits).
    - `alpha` (float): Weighting factor for the positive class (e.g., 0.25).
    - `gamma` (float): Focusing parameter (e.g., 2.0).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Use reduction='none' to apply focal term and alpha before reducing
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # logits: [N, ...], raw model output
        # targets: [N, ...], float labels
        
        loss_bce = self.bce_loss(logits, targets)
        
        # p_t = p if y=1, 1-p if y=0
        # p = sigmoid(logits)
        # p_t = exp(-BCE_loss)
        pt = torch.exp(-loss_bce)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1.0 - pt).pow(self.gamma)
        
        loss = focal_term * loss_bce
        
        if self.alpha is not None:
            # Alpha term: alpha if y=1, (1-alpha) if y=0
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss, compatible with `CrossEntropyLoss` (expects raw logits).
    - `alpha` (torch.Tensor): A tensor of weights for each class [C].
    - `gamma` (float): Focusing parameter (e.g., 2.0).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # [C] tensor of weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N]
        
        # Calculate cross-entropy loss without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # pt = p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1.0 - pt).pow(self.gamma)
        
        loss = focal_term * ce_loss
        
        if self.alpha is not None:
            # Move alpha to the correct device if it's not already
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha).to(logits.device)
            elif self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
                
            # Gather the weights for the target classes
            alpha_t = self.alpha.gather(0, targets) # [N]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ------------------------------
# GatedGCNConv
# ------------------------------
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

# ------------------------------
# RWSE embeddings
# ------------------------------
def compute_rw_diag(data, steps: int):
    N = data.num_nodes
    if N == 0:
        return torch.zeros((0, steps+1), dtype=torch.float)
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

# ------------------------------
# GPS Model
# ------------------------------
class GPS(nn.Module):
    def __init__(self, in_dim, channels, pe_dim, num_layers, num_classes, conv_type="gated", edge_dim=0, pool="mean"):
        super().__init__()
        node_out_dim = channels - pe_dim if pe_dim > 0 else channels
        if node_out_dim <= 0: raise ValueError("channels must be > pe_dim")
        self.node_emb = nn.Linear(in_dim, node_out_dim)
        self.pe_lin = nn.Linear(pe_dim, pe_dim) if pe_dim>0 else None
        self.convs = nn.ModuleList()
        
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Pooling {pool} not supported.")
        
        for _ in range(num_layers):
            if conv_type=="gated":
                mpnn = GatedGCNConv(channels, channels, edge_dim=edge_dim)
                layer = GPSConv(channels, mpnn, heads=4)
            else:
                nn_seq = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))
                layer = GPSConv(channels, GINEConv(nn_seq, edge_dim=edge_dim), heads=4, attn_dropout=0.5)
            self.convs.append(layer)
        self.lin_out = nn.Linear(channels, num_classes)

    def forward(self, x, x_pe, edge_index, edge_attr, batch):
        if x is None:
             x = torch.ones((batch.size(0), 1), device=edge_index.device)
        if x.dim()==1: x=x.view(-1,1)
        x = x.float()
        if self.pe_lin is not None and x_pe is not None:
            x = torch.cat([self.node_emb(x), self.pe_lin(x_pe.float())], dim=1)
        else:
            x = self.node_emb(x)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = self.pool(x, batch)
        x = F.dropout(x, p=0.05, training=self.training)
        return self.lin_out(x)

# ------------------------------
# Train/Eval Functions
# ------------------------------
def train_epoch(model, loader, optimizer, device, criterion, num_classes):
    model.train(); total_loss=0.0
    for data in tqdm(loader, leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, getattr(data,"pe",None), data.edge_index, data.edge_attr, data.batch)
        if num_classes == 2:
            target = data.y.view(-1).float()
            out = out.view(-1)
        else:
            target = data.y.view(-1)
        loss = criterion(out, target)
        loss.backward(); optimizer.step()
        total_loss += loss.item()*data.num_graphs
    return total_loss/len(loader.dataset)

@torch.no_grad()
def predict(model, loader, device, num_classes):
    model.eval(); y_true_list, y_prob_list=[],[]
    for data in tqdm(loader, leave=False):
        data = data.to(device)
        out = model(data.x, getattr(data,"pe",None), data.edge_index, data.edge_attr, data.batch)
        y_true_list.append(data.y.view(-1,1).cpu())
        if num_classes == 2:
            y_prob_list.append(torch.sigmoid(out.view(-1,1)).cpu())
        else:
            y_prob_list.append(F.softmax(out, dim=1).cpu())
    return torch.cat(y_true_list,0), torch.cat(y_prob_list,0)

def compute_optimal_threshold(y_true, y_prob):
    thresholds = torch.arange(0.01, 1.01, 0.01)
    f1_scores=[]
    for t in thresholds:
        f1_scores.append(f1_score(y_true, (y_prob>=t).int(), average='macro'))
    return thresholds[torch.argmax(torch.tensor(f1_scores))].item()

def compute_metrics(y_true, y_prob, num_classes, optimal_threshold=None):
    if num_classes == 2:
        if optimal_threshold is None: optimal_threshold=0.5
        y_pred = (y_prob>=optimal_threshold).int()
    else:
        y_pred = torch.argmax(y_prob, dim=1)
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'balacc': balanced_accuracy_score(y_true, y_pred)
    }
    try:
        if num_classes == 2:
            metrics['rocauc'] = roc_auc_score(y_true, y_prob)
            metric = BinaryAUPRC()
            metric.update(y_prob.squeeze(-1), y_true.squeeze(-1))
            metrics['prauc'] = metric.compute().item()
        else:
            metrics['rocauc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metric = MulticlassAUPRC(num_classes=num_classes)
            metric.update(y_prob, y_true.squeeze())
            metrics['prauc'] = metric.compute().item()
    except ValueError as e:
        print(f"Could not compute AUC score (likely due to single-class batch): {e}")
        metrics['rocauc'] = 0.0
        metrics['prauc'] = 0.0
    return metrics, y_pred

# ------------------------------
# Main
# ------------------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv", help="Dataset name")
    parser.add_argument("--conv_type", type=str, default="gated", choices=["gine","gated"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rw_steps", type=int, default=16)
    parser.add_argument("--rwse_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--metric", type=str, default="rocauc", choices=["acc","balacc","rocauc","prauc"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean","sum"], help="Pooling type for graph-level readout")
    
    # --- ADDED: Arguments for Loss Functions ---
    parser.add_argument('--use_class_weight', action='store_true',
                        help='Use inverse class frequency to weight the loss function.')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of standard NLL/BCE Loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma focusing parameter for Focal Loss.')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha balancing parameter for Binary Focal Loss.')
    
    args=parser.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- MODIFIED: Class Weight Calculation Variables ---
    train_counts = None
    pos_weight_tensor = None
    class_weights_tensor = None
    actual_rwse_dim = 0 # Initialize

    # Load Dataset
    if "ogbg-molhiv" in args.dataset:
        dataset = PygGraphPropPredDataset(name=args.dataset)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        num_classes = 2
        
        # --- ADDED: Calculate weights for OGB ---
        if args.use_class_weight:
            y_train = dataset.y[train_idx].squeeze()
            pos_samples = (y_train == 1).sum().item()
            neg_samples = (y_train == 0).sum().item()
            if pos_samples == 0: pos_samples = 1e-8
            pos_weight_tensor = torch.tensor([neg_samples / pos_samples], device=device)
            print(f"OGB dataset. Using class weights. pos_weight = {pos_weight_tensor.item():.4f}")

    else:  # TU datasets
        dataset = TUDataset(root=f"./data/TUDataset", name=args.dataset)
        num_classes = dataset.num_classes
        
        # ===================================================================
        # START: Patched Splitting Logic (replacing StratifiedShuffleSplit)
        # ===================================================================
        
        # Create a list of all indices from the dataset and shuffle them
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        # Define split sizes for an 80/10/10 random split
        test_size = len(dataset) // 10
        valid_size = len(dataset) // 10
        
        # Generate the final index lists for train, validation, and test sets
        test_idx = indices[:test_size]
        valid_idx = indices[test_size : test_size + valid_size]
        train_idx = indices[test_size + valid_size :]
        # ===================================================================
        # END: Patched Splitting Logic
        # ===================================================================
        
        # --------------------------
        # Compute RWSE (This part remains unchanged)
        # --------------------------
        in_rw_dim = args.rw_steps + 1
        actual_rwse_dim = args.rwse_dim if args.rwse_dim else in_rw_dim
        rw_proj = nn.Linear(in_rw_dim, actual_rwse_dim, bias=False)
        nn.init.orthogonal_(rw_proj.weight)
        rw_proj.eval()
        
        # Assign RWSE
        data_list = []
        for i in tqdm(range(len(dataset)), desc="Computing RWSE"):
            data = dataset[i]
            diag = compute_rw_diag(data, steps=args.rw_steps)
            with torch.no_grad():
                data.pe = rw_proj(diag)
            data_list.append(data)
        
        # --------------------------
        # Split (This part remains unchanged and now uses the randomly generated indices)
        # --------------------------
        train_list = [data_list[i] for i in train_idx]
        valid_list = [data_list[i] for i in valid_idx]
        test_list = [data_list[i] for i in test_idx]

        train_labels = [data.y.item() for data in train_list]
        train_counts = Counter(train_labels)
        
        # Get labels and counts for the validation set
        valid_labels = [data.y.item() for data in valid_list]
        valid_counts = Counter(valid_labels)
        
        # Get labels and counts for the test set
        test_labels = [data.y.item() for data in test_list]
        test_counts = Counter(test_labels)
        
        print("\n--- Class Distribution ---")
        print(f"Training samples: {len(train_list)}")
        for cls, count in sorted(train_counts.items()):
            print(f"  Class {cls}: {count} samples")
        
        print(f"\nValidation samples: {len(valid_list)}")
        for cls, count in sorted(valid_counts.items()):
            print(f"  Class {cls}: {count} samples")
        
        print(f"\nTest samples: {len(test_list)}")
        for cls, count in sorted(test_counts.items()):
            print(f"  Class {cls}: {count} samples")
        print("--------------------------\n")

        # --- ADDED: Calculate weights for TU Datasets ---
        if args.use_class_weight:
            if num_classes == 2:
                pos_samples = train_counts.get(1, 0)
                neg_samples = train_counts.get(0, 0)
                if pos_samples == 0: pos_samples = 1e-8
                pos_weight_tensor = torch.tensor([neg_samples / pos_samples], device=device)
                print(f"TU dataset. Using class weights. pos_weight = {pos_weight_tensor.item():.4f}")
            else:
                total_samples = len(train_idx)
                class_weights_tensor = torch.zeros(num_classes)
                for cls, count in train_counts.items():
                    if count == 0: count = 1e-8
                    class_weights_tensor[cls] = total_samples / (num_classes * count)
                class_weights_tensor = class_weights_tensor.to(device)
                print(f"TU dataset. Using class weights: {class_weights_tensor}")
        
        train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_list, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)

    in_dim = dataset.num_node_features
    if in_dim == 0: in_dim = 1
    edge_dim = dataset.num_edge_features if dataset[0].edge_attr is not None else 0
    channels = args.channels
    if actual_rwse_dim >= channels: channels = actual_rwse_dim+1
    
    # --- MODIFIED: Folder Naming ---
    loss_str = ""
    if args.use_class_weight:
        loss_str = "_class_weight"
    elif args.use_focal_loss:
        loss_str = f"_focal_g{args.focal_gamma}_a{args.focal_alpha}"

    folder_name = f"pool_{args.pool}_metric{args.metric}_conv{args.conv_type}_layers{args.num_layers}_ch{channels}_rwse{actual_rwse_dim}_lr{args.lr}_bs{args.batch_size}{loss_str}"
    
    # --- MODIFIED: Base Results Directory ---
    result_dir = os.path.join("results_weighted", args.dataset, "GPS", folder_name)
    os.makedirs(result_dir, exist_ok=True)

    # Multi-run loop
    for run in range(1,args.runs+1):
        print(f"\nRun {run}/{args.runs}")
        run_dir = os.path.join(result_dir,f"run_{run}")
        os.makedirs(run_dir,exist_ok=True)

        model = GPS(in_dim=in_dim, channels=channels, pe_dim=actual_rwse_dim,
                    num_layers=args.num_layers, conv_type=args.conv_type, edge_dim=edge_dim, 
                    num_classes=num_classes, pool=args.pool).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # --- MODIFIED: Criterion Selection ---
        if num_classes == 2: # Binary
            if args.use_focal_loss:
                print(f"Using BinaryFocalLoss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
                criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
            else:
                print(f"Using BCEWithLogitsLoss (pos_weight={pos_weight_tensor is not None})")
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else: # Multiclass
            if args.use_focal_loss:
                print(f"Using FocalLoss (gamma={args.focal_gamma}, alpha={'class_weights' if class_weights_tensor is not None else 'None'})")
                criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
            else:
                print(f"Using CrossEntropyLoss (weight={class_weights_tensor is not None})")
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # --- End of MODIFIED block ---
        
        # Define the warmup scheduler using a lambda function
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch+1)/args.warmup_epochs))

        best_val_metric = -np.inf
        best_model_path = os.path.join(run_dir, "best_model.pt")

        for epoch in range(1,args.epochs+1):
            loss = train_epoch(model, train_loader, optimizer, device, criterion, num_classes)
            y_val_true, y_val_prob = predict(model, valid_loader, device, num_classes)
            opt_thresh = compute_optimal_threshold(y_val_true, y_val_prob) if num_classes == 2 else None
            metrics, _ = compute_metrics(y_val_true, y_val_prob, num_classes, optimal_threshold=opt_thresh)
            val_metric = metrics[args.metric]

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), best_model_path)

            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val {args.metric} {val_metric:.4f}")
            
            # THE CRITICAL FIX: Step the scheduler to update the learning rate
            scheduler.step()
        
        # Evaluation and summary logic follows...
        print("Evaluating best model on test set...")
        model.load_state_dict(torch.load(best_model_path))
        y_true_test, y_prob_test = predict(model, test_loader, device, num_classes)
        if num_classes == 2:
            y_true_val, y_prob_val = predict(model, valid_loader, device, num_classes)
            opt_thresh = compute_optimal_threshold(y_true_val, y_prob_val)
        else:
            opt_thresh = None
        metrics_default, y_pred_def = compute_metrics(y_true_test, y_prob_test, num_classes)
        metrics_opt, y_pred_opt = compute_metrics(y_true_test, y_prob_test, num_classes, optimal_threshold=opt_thresh)
        run_results = {
            "best_validation_metric": best_val_metric,
            "accuracy": metrics_opt['acc'],
            "balanced_accuracy": metrics_opt['balacc'],
            "auc_roc": metrics_opt['rocauc'],
            "auc_pr": metrics_opt['prauc'],
            "classification_report_default": classification_report(y_true_test, y_pred_def, zero_division=0, output_dict=True),
            "classification_report_optimal": classification_report(y_true_test, y_pred_opt, zero_division=0, output_dict=True)
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(run_results, f, indent=4)
        print(f"Run {run} results saved at {run_dir}")

    # Summary logic
    print("\nSummarizing results across all runs...")
    summary_dir = os.path.join(result_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    all_best_val_metrics = []
    all_test_metrics = {'accuracy': [],'balanced_accuracy': [],'auc_roc': [],'auc_pr': [],}
    all_reports_default = []
    all_reports_optimal = []
    for run in range(1, args.runs + 1):
        run_metrics_path = os.path.join(result_dir, f"run_{run}", "metrics.json")
        if os.path.exists(run_metrics_path):
            with open(run_metrics_path, "r") as f:
                run_metrics = json.load(f)
            all_best_val_metrics.append(run_metrics["best_validation_metric"])
            for key in all_test_metrics.keys():
                all_test_metrics[key].append(run_metrics[key])
            all_reports_default.append(run_metrics["classification_report_default"])
            all_reports_optimal.append(run_metrics["classification_report_optimal"])
    summary_output = {
        "metric": args.metric,
        "best_valid_mean": np.mean(all_best_val_metrics) if all_best_val_metrics else 0.0,
        "best_valid_std": np.std(all_best_val_metrics) if all_best_val_metrics else 0.0,
    }
    for key, values in all_test_metrics.items():
        summary_output[f"{key}_mean"] = np.mean(values) if values else 0.0
        summary_output[f"{key}_std"] = np.std(values) if values else 0.0
    with open(os.path.join(summary_dir, "summary.json"), "w") as f:
        json.dump(summary_output, f, indent=4)
    with open(os.path.join(summary_dir, "reports_default.json"), "w") as f:
        json.dump(all_reports_default, f, indent=4)
    with open(os.path.join(summary_dir, "reports_optimal.json"), "w") as f:
        json.dump(all_reports_optimal, f, indent=4)
    print(f"Summary reports saved in {summary_dir}")