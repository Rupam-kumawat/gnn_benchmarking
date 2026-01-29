


#THIS WORKS FOR COX2 AND MOLHIV
#!/usr/bin/env python3
import argparse, importlib, os, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
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
        x = F.dropout(x, p=0.05, training=self.training)
        return self.lin_out(x)

# ------------------------------
# Train/Eval Functions
# ------------------------------
def train_epoch(model, loader, optimizer, device, criterion):
    model.train(); total_loss=0.0
    for data in tqdm(loader, leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, getattr(data,"pe",None), data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.view(-1), data.y.view(-1).float())
        loss.backward(); optimizer.step()
        total_loss += loss.item()*data.num_graphs
    return total_loss/len(loader.dataset)

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); y_true_list, y_pred_list=[],[]
    for data in tqdm(loader, leave=False):
        data = data.to(device)
        out = model(data.x, getattr(data,"pe",None), data.edge_index, data.edge_attr, data.batch)
        y_true_list.append(data.y.view(-1,1).cpu())
        y_pred_list.append(torch.sigmoid(out.view(-1,1)).cpu())
    return torch.cat(y_true_list,0), torch.cat(y_pred_list,0)

def compute_optimal_threshold(y_true, y_prob):
    thresholds = torch.arange(0.01, 1.01, 0.01)
    f1_scores=[]
    for t in thresholds:
        f1_scores.append(f1_score(y_true, (y_prob>=t).int(), average='macro'))
    return thresholds[torch.argmax(torch.tensor(f1_scores))].item()

def compute_metrics(y_true, y_prob, optimal_threshold=None, num_classes=2):
    if optimal_threshold is None: optimal_threshold=0.5
    y_pred_bin = (y_prob>=optimal_threshold).int()
    metrics = {}
    metrics['acc']=accuracy_score(y_true, y_pred_bin)
    metrics['balacc']=balanced_accuracy_score(y_true, y_pred_bin)
    metrics['rocauc']=roc_auc_score(y_true, y_prob)
    if num_classes == 2:  # binary
        metric = BinaryAUPRC()
        metric.update(y_prob.squeeze(-1), y_true.squeeze(-1))
    else:  # multi-class
        metric = MulticlassAUPRC(num_classes=num_classes)
    
        metric.update(y_prob, y_true)
    metrics['prauc'] = metric.compute().item()
    return metrics, y_pred_bin

# ------------------------------
# Main
# ------------------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv", help="Dataset name")
    parser.add_argument("--conv_type", type=str, default="gated", choices=["gine","gated"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rw_steps", type=int, default=16)
    parser.add_argument("--rwse_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--metric", type=str, default="rocauc", choices=["acc","balacc","rocauc","prauc"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean","sum"], help="Pooling type for graph-level readout")
    
    args=parser.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------
    # Load Dataset
    # --------------------------
    if args.dataset in ["ogbg-molhiv"]:
        dataset = PygGraphPropPredDataset(root=f"./data/OGB",name=args.dataset)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        y_all = dataset.data.y.view(-1).numpy()

    else:  # TU datasets
        dataset = TUDataset(root=f"./data/TUDataset", name=args.dataset)
    
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

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_list, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)
    # else:  # TU datasets
        
    #     # dataset = TUDataset(root=f"./data/TU_{args.dataset}", name=args.dataset)
    #     dataset = TUDataset(root=f"./data/TUDataset", name=args.dataset)
    #     y_all = np.array([data.y.item() if data.y.dim()==0 else data.y.numpy() for data in dataset])
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    #     train_idx, test_idx = next(sss.split(np.zeros(len(y_all)), y_all))
    #     sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    #     valid_idx, test_idx = next(sss2.split(np.zeros(len(test_idx)), y_all[test_idx]))

        

    # # --------------------------
    # # Compute RWSE
    # # --------------------------
    # in_rw_dim = args.rw_steps+1
    # actual_rwse_dim = args.rwse_dim if args.rwse_dim else in_rw_dim
    # rw_proj = nn.Linear(in_rw_dim, actual_rwse_dim, bias=False)
    # nn.init.orthogonal_(rw_proj.weight)
    # rw_proj.eval()

    # # Assign RWSE
    # data_list=[]
    # for i in tqdm(range(len(dataset)), desc="Computing RWSE"):
    #     data = dataset[i]
    #     diag = compute_rw_diag(data, steps=args.rw_steps)
    #     with torch.no_grad(): data.pe = rw_proj(diag)
    #     data_list.append(data)

    # # --------------------------
    # # Split
    # # --------------------------
    # train_list = [data_list[i] for i in train_idx]
    # valid_list = [data_list[i] for i in valid_idx]
    # test_list = [data_list[i] for i in test_idx]
    # train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_list, batch_size=args.batch_size, shuffle=False)
    # test_loader  = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)

    # Node/edge dims
    in_dim = dataset.num_node_features
    if in_dim == 0:
        in_dim = 1
        for data in data_list:
            if data.x is None:
                data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    edge_dim = dataset.num_edge_features if hasattr(dataset[0], 'edge_attr') else 0
    channels = args.channels
    if actual_rwse_dim >= channels: channels = actual_rwse_dim+1

    # Results folder
    # folder_name = f"metric_{args.metric}_conv{args.conv_type}_layers{args.num_layers}_ch{channels}_rwse{actual_rwse_dim}_lr{args.lr}_bs{args.batch_size}"
    folder_name = f"pool_{args.pool}_metric{args.metric}_conv{args.conv_type}_layers{args.num_layers}_ch{channels}_rwse{actual_rwse_dim}_lr{args.lr}_bs{args.batch_size}"

    result_dir = os.path.join("results", args.dataset, "GPS", folder_name)
    os.makedirs(result_dir, exist_ok=True)

    # --------------------------
    # Multi-run loop
    # --------------------------
    for run in range(1,args.runs+1):
        print(f"\nRun {run}/{args.runs}")
        run_dir = os.path.join(result_dir,f"run_{run}")
        os.makedirs(run_dir,exist_ok=True)

        model = GPS(in_dim=in_dim, channels=channels, pe_dim=actual_rwse_dim,
                    num_layers=args.num_layers, conv_type=args.conv_type, edge_dim=edge_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss() if args.dataset=="ogbg-molhiv" else nn.CrossEntropyLoss()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0,e/args.warmup_epochs))

        best_val_metric = -np.inf
        best_model_path = os.path.join(run_dir, "best_model.pt")

        for epoch in range(1,args.epochs+1):
            loss = train_epoch(model, train_loader, optimizer, device, criterion)
            y_val, y_pred_val = predict(model, valid_loader, device)
            opt_thresh = compute_optimal_threshold(y_val, y_pred_val)
            metrics, _ = compute_metrics(y_val, y_pred_val, optimal_threshold=opt_thresh)
            val_metric = metrics[args.metric]

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), best_model_path)

            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val {args.metric} {val_metric:.4f}")
            scheduler.step()
        
        # -----------------------------------------------------
        # START: MODIFIED RESULTS SAVING LOGIC FOR A SINGLE RUN
        # -----------------------------------------------------
        # Evaluate best model on the test set
        print("Evaluating best model on test set...")
        model.load_state_dict(torch.load(best_model_path))
        
        # Get predictions for test set
        y_true_test, y_prob_test = predict(model, test_loader, device)
        
        # Determine optimal threshold from validation set to avoid data leakage
        y_true_val, y_prob_val = predict(model, valid_loader, device)
        opt_thresh = compute_optimal_threshold(y_true_val, y_prob_val)

        # Compute metrics for test set using both default and optimal thresholds
        metrics_default, y_pred_def = compute_metrics(y_true_test, y_prob_test)
        metrics_opt, y_pred_opt = compute_metrics(y_true_test, y_prob_test, optimal_threshold=opt_thresh)
        
        # Structure the results for this run as per metrics.json
        run_results = {
            "best_validation_metric": best_val_metric,
            "accuracy": metrics_opt['acc'],
            "balanced_accuracy": metrics_opt['balacc'],
            "auc_roc": metrics_opt['rocauc'],
            "auc_pr": metrics_opt['prauc'],
            "classification_report_default": classification_report(y_true_test, y_pred_def, zero_division=0, output_dict=True),
            "classification_report_optimal": classification_report(y_true_test, y_pred_opt, zero_division=0, output_dict=True)
        }

        # Save the structured results to metrics.json
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(run_results, f, indent=4)
        print(f"Run {run} results saved at {run_dir}")
        # ---------------------------------------------------
        # END: MODIFIED RESULTS SAVING LOGIC FOR A SINGLE RUN
        # ---------------------------------------------------

    # -------------------------------------
    # START: MODIFIED SUMMARY SAVING LOGIC
    # -------------------------------------
    print("\nSummarizing results across all runs...")
    summary_dir = os.path.join(result_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    all_best_val_metrics = []
    all_test_metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'auc_roc': [],
        'auc_pr': [],
    }
    all_reports_default = []
    all_reports_optimal = []
    
    # Collect results from each run's metrics.json
    for run in range(1, args.runs + 1):
        run_metrics_path = os.path.join(result_dir, f"run_{run}", "metrics.json")
        with open(run_metrics_path, "r") as f:
            run_metrics = json.load(f)
        
        all_best_val_metrics.append(run_metrics["best_validation_metric"])
        
        for key in all_test_metrics.keys():
            all_test_metrics[key].append(run_metrics[key])

        all_reports_default.append(run_metrics["classification_report_default"])
        all_reports_optimal.append(run_metrics["classification_report_optimal"])
        
    # Create and save summary.json with mean and std dev
    summary_output = {
        "metric": args.metric,
        "best_valid_mean": np.mean(all_best_val_metrics),
        "best_valid_std": np.std(all_best_val_metrics),
    }
    for key, values in all_test_metrics.items():
        summary_output[f"{key}_mean"] = np.mean(values)
        summary_output[f"{key}_std"] = np.std(values)

    with open(os.path.join(summary_dir, "summary.json"), "w") as f:
        json.dump(summary_output, f, indent=4)

    # Create and save the collected classification reports
    with open(os.path.join(summary_dir, "reports_default.json"), "w") as f:
        json.dump(all_reports_default, f, indent=4)
    
    with open(os.path.join(summary_dir, "reports_optimal.json"), "w") as f:
        json.dump(all_reports_optimal, f, indent=4)
        
    print(f"Summary reports saved in {summary_dir}")
    # -----------------------------------
    # END: MODIFIED SUMMARY SAVING LOGIC
    # -----------------------------------











# #THIS WORKS FOR COX2 AND MOLHIV
# #!/usr/bin/env python3
# import argparse, importlib, os, itertools
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool, global_add_pool
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
# from torch_geometric.nn.conv import MessagePassing
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
# from torch.optim.lr_scheduler import LambdaLR
# from tqdm import tqdm
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
# from torcheval.metrics import BinaryAUPRC, MulticlassAUPRC
# from torch_geometric.datasets import TUDataset
# import json
# from collections import Counter

# import random
# random.seed(42) # For reproducibility

# # ------------------------------
# # Safe globals patch for torch.load (PyTorch >= 2.6)
# # ------------------------------
# if torch.__version__ >= "2.6.0":
#     import torch_geometric.data
#     from torch_geometric.data import Data, Batch
#     safe_globals = [Data, Batch]
#     optional_classes = ["DataEdgeAttr", "DataTensorAttr", "GlobalStorage"]
#     for cls_name in optional_classes:
#         try:
#             cls = getattr(torch_geometric.data, cls_name, None)
#             if cls is None:
#                 for submod in ["torch_geometric.data.data", "torch_geometric.data.storage"]:
#                     try:
#                         mod = importlib.import_module(submod)
#                         cls = getattr(mod, cls_name, None)
#                         if cls is not None:
#                             break
#                     except ImportError:
#                         pass
#             if cls is not None:
#                 safe_globals.append(cls)
#         except Exception:
#             pass
#     torch.serialization.add_safe_globals(safe_globals)

# # ------------------------------
# # GatedGCNConv
# # ------------------------------
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

# # ------------------------------
# # RWSE embeddings
# # ------------------------------
# def compute_rw_diag(data, steps: int):
#     N = data.num_nodes
#     if N == 0:
#         return torch.zeros((0, steps+1), dtype=torch.float)
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

# # ------------------------------
# # GPS Model
# # ------------------------------
# class GPS(nn.Module):
#     def __init__(self, in_dim, channels, pe_dim, num_layers, conv_type="gated", edge_dim=0, pool="mean"):
#         super().__init__()
#         node_out_dim = channels - pe_dim if pe_dim > 0 else channels
#         if node_out_dim <= 0: raise ValueError("channels must be > pe_dim")
#         self.node_emb = nn.Linear(in_dim, node_out_dim)
#         self.pe_lin = nn.Linear(pe_dim, pe_dim) if pe_dim>0 else None
#         self.convs = nn.ModuleList()
#         self.pool=pool
#         for _ in range(num_layers):
#             if conv_type=="gated":
#                 mpnn = GatedGCNConv(channels, channels, edge_dim=edge_dim)
#                 layer = GPSConv(channels, mpnn, heads=4)
#             else:
#                 nn_seq = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))
#                 layer = GPSConv(channels, GINEConv(nn_seq, edge_dim=edge_dim), heads=4, attn_dropout=0.5)
#             self.convs.append(layer)
#         self.lin_out = nn.Linear(channels, 1)

#     def forward(self, x, x_pe, edge_index, edge_attr, batch):
#         x = x if x is not None else torch.ones((edge_index.max().item()+1, 1), device=edge_index.device)

#         if x.dim()==1: x=x.view(-1,1)
#         x = x.float()
#         if self.pe_lin is not None and x_pe is not None:
#             x = torch.cat([self.node_emb(x), self.pe_lin(x_pe.float())], dim=1)
#         else:
#             x = self.node_emb(x)
#         for conv in self.convs:
#             x = conv(x, edge_index, batch, edge_attr=edge_attr)
#         if self.pool=="mean":
#             x = global_mean_pool(x, batch)
#         elif self.pool=="sum":
#             x = global_add_pool(x, batch)
#         x = F.dropout(x, p=0.05, training=self.training)
#         return self.lin_out(x)

# # ------------------------------
# # Train/Eval Functions
# # ------------------------------
# def train_epoch(model, loader, optimizer, device, criterion):
#     model.train(); total_loss=0.0
#     for data in tqdm(loader, leave=False):
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, getattr(data,"pe",None), data.edge_index, data.edge_attr, data.batch)
#         loss = criterion(out.view(-1), data.y.view(-1).float())
#         loss.backward(); optimizer.step()
#         total_loss += loss.item()*data.num_graphs
#     return total_loss/len(loader.dataset)

# @torch.no_grad()
# def predict(model, loader, device):
#     model.eval(); y_true_list, y_pred_list=[],[]
#     for data in tqdm(loader, leave=False):
#         data = data.to(device)
#         out = model(data.x, getattr(data,"pe",None), data.edge_index, data.edge_attr, data.batch)
#         y_true_list.append(data.y.view(-1,1).cpu())
#         y_pred_list.append(torch.sigmoid(out.view(-1,1)).cpu())
#     return torch.cat(y_true_list,0), torch.cat(y_pred_list,0)

# def compute_optimal_threshold(y_true, y_prob):
#     thresholds = torch.arange(0.01, 1.01, 0.01)
#     f1_scores=[]
#     for t in thresholds:
#         f1_scores.append(f1_score(y_true, (y_prob>=t).int(), average='macro'))
#     return thresholds[torch.argmax(torch.tensor(f1_scores))].item()

# def compute_metrics(y_true, y_prob, optimal_threshold=None, num_classes=2):
#     if optimal_threshold is None: optimal_threshold=0.5
#     y_pred_bin = (y_prob>=optimal_threshold).int()
#     metrics = {}
#     metrics['acc']=accuracy_score(y_true, y_pred_bin)
#     metrics['balacc']=balanced_accuracy_score(y_true, y_pred_bin)
#     metrics['rocauc']=roc_auc_score(y_true, y_prob)
#     if num_classes == 2:  # binary
#         metric = BinaryAUPRC()
#         metric.update(y_prob.squeeze(-1), y_true.squeeze(-1))
#     else:  # multi-class
#         metric = MulticlassAUPRC(num_classes=num_classes)
    
#         metric.update(y_prob, y_true)
#     metrics['prauc'] = metric.compute().item()
#     return metrics, y_pred_bin

# # ------------------------------
# # Main
# # ------------------------------
# if __name__=="__main__":
#     parser=argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default="ogbg-molhiv", help="Dataset name")
#     parser.add_argument("--conv_type", type=str, default="gated", choices=["gine","gated"])
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--rw_steps", type=int, default=16)
#     parser.add_argument("--rwse_dim", type=int, default=None)
#     parser.add_argument("--num_layers", type=int, default=10)
#     parser.add_argument("--channels", type=int, default=64)
#     parser.add_argument("--warmup_epochs", type=int, default=5)
#     parser.add_argument("--metric", type=str, default="rocauc", choices=["acc","balacc","rocauc","prauc"])
#     parser.add_argument("--runs", type=int, default=3)
#     parser.add_argument("--pool", type=str, default="mean", choices=["mean","sum"], help="Pooling type for graph-level readout")
    
#     args=parser.parse_args()

#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # --------------------------
#     # Load Dataset
#     # --------------------------
#     if args.dataset in ["ogbg-molhiv"]:
#         dataset = PygGraphPropPredDataset(root=f"./data/OGB",name=args.dataset)
#         split_idx = dataset.get_idx_split()
#         train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#         y_all = dataset.data.y.view(-1).numpy()

#     else:  # TU datasets
#         dataset = TUDataset(root=f"./data/TUDataset", name=args.dataset)
    
#         # ===================================================================
#         # START: Patched Splitting Logic (replacing StratifiedShuffleSplit)
#         # ===================================================================
        
    
#         # Create a list of all indices from the dataset and shuffle them
#         indices = list(range(len(dataset)))
#         random.shuffle(indices)
    
#         # Define split sizes for an 80/10/10 random split
#         test_size = len(dataset) // 10
#         valid_size = len(dataset) // 10
    
#         # Generate the final index lists for train, validation, and test sets
#         test_idx = indices[:test_size]
#         valid_idx = indices[test_size : test_size + valid_size]
#         train_idx = indices[test_size + valid_size :]
#         # ===================================================================
#         # END: Patched Splitting Logic
#         # ===================================================================
    
#         # --------------------------
#         # Compute RWSE (This part remains unchanged)
#         # --------------------------
#         in_rw_dim = args.rw_steps + 1
#         actual_rwse_dim = args.rwse_dim if args.rwse_dim else in_rw_dim
#         rw_proj = nn.Linear(in_rw_dim, actual_rwse_dim, bias=False)
#         nn.init.orthogonal_(rw_proj.weight)
#         rw_proj.eval()
    
#         # Assign RWSE
#         data_list = []
#         for i in tqdm(range(len(dataset)), desc="Computing RWSE"):
#             data = dataset[i]
#             diag = compute_rw_diag(data, steps=args.rw_steps)
#             with torch.no_grad():
#                 data.pe = rw_proj(diag)
#             data_list.append(data)
    
#         # --------------------------
#         # Split (This part remains unchanged and now uses the randomly generated indices)
#         # --------------------------
#         train_list = [data_list[i] for i in train_idx]
#         valid_list = [data_list[i] for i in valid_idx]
#         test_list = [data_list[i] for i in test_idx]

#         train_labels = [data.y.item() for data in train_list]
#         train_counts = Counter(train_labels)
        
#         # Get labels and counts for the validation set
#         valid_labels = [data.y.item() for data in valid_list]
#         valid_counts = Counter(valid_labels)
        
#         # Get labels and counts for the test set
#         test_labels = [data.y.item() for data in test_list]
#         test_counts = Counter(test_labels)
        
#         print("\n--- Class Distribution ---")
#         print(f"Training samples: {len(train_list)}")
#         for cls, count in sorted(train_counts.items()):
#             print(f"  Class {cls}: {count} samples")
        
#         print(f"\nValidation samples: {len(valid_list)}")
#         for cls, count in sorted(valid_counts.items()):
#             print(f"  Class {cls}: {count} samples")
        
#         print(f"\nTest samples: {len(test_list)}")
#         for cls, count in sorted(test_counts.items()):
#             print(f"  Class {cls}: {count} samples")
#         print("--------------------------\n")
    
#         train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
#         valid_loader = DataLoader(valid_list, batch_size=args.batch_size, shuffle=False)
#         test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)
#     # else:  # TU datasets
        
#     #     # dataset = TUDataset(root=f"./data/TU_{args.dataset}", name=args.dataset)
#     #     dataset = TUDataset(root=f"./data/TUDataset", name=args.dataset)
#     #     y_all = np.array([data.y.item() if data.y.dim()==0 else data.y.numpy() for data in dataset])
#     #     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     #     train_idx, test_idx = next(sss.split(np.zeros(len(y_all)), y_all))
#     #     sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
#     #     valid_idx, test_idx = next(sss2.split(np.zeros(len(test_idx)), y_all[test_idx]))

        

#     # # --------------------------
#     # # Compute RWSE
#     # # --------------------------
#     # in_rw_dim = args.rw_steps+1
#     # actual_rwse_dim = args.rwse_dim if args.rwse_dim else in_rw_dim
#     # rw_proj = nn.Linear(in_rw_dim, actual_rwse_dim, bias=False)
#     # nn.init.orthogonal_(rw_proj.weight)
#     # rw_proj.eval()

#     # # Assign RWSE
#     # data_list=[]
#     # for i in tqdm(range(len(dataset)), desc="Computing RWSE"):
#     #     data = dataset[i]
#     #     diag = compute_rw_diag(data, steps=args.rw_steps)
#     #     with torch.no_grad(): data.pe = rw_proj(diag)
#     #     data_list.append(data)

#     # # --------------------------
#     # # Split
#     # # --------------------------
#     # train_list = [data_list[i] for i in train_idx]
#     # valid_list = [data_list[i] for i in valid_idx]
#     # test_list = [data_list[i] for i in test_idx]
#     # train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
#     # valid_loader = DataLoader(valid_list, batch_size=args.batch_size, shuffle=False)
#     # test_loader  = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)

#     # Node/edge dims
#     in_dim = dataset.num_node_features
#     if in_dim == 0:
#         in_dim = 1
#         for data in data_list:
#             if data.x is None:
#                 data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
#     edge_dim = dataset.num_edge_features if hasattr(dataset[0], 'edge_attr') else 0
#     channels = args.channels
#     if actual_rwse_dim >= channels: channels = actual_rwse_dim+1

#     # Results folder
#     # folder_name = f"metric_{args.metric}_conv{args.conv_type}_layers{args.num_layers}_ch{channels}_rwse{actual_rwse_dim}_lr{args.lr}_bs{args.batch_size}"
#     folder_name = f"pool_{args.pool}_metric{args.metric}_conv{args.conv_type}_layers{args.num_layers}_ch{channels}_rwse{actual_rwse_dim}_lr{args.lr}_bs{args.batch_size}"

#     result_dir = os.path.join("results", args.dataset, "GPS", folder_name)
#     os.makedirs(result_dir, exist_ok=True)

#     # --------------------------
#     # Multi-run loop
#     # --------------------------
#     for run in range(1,args.runs+1):
#         print(f"\nRun {run}/{args.runs}")
#         run_dir = os.path.join(result_dir,f"run_{run}")
#         os.makedirs(run_dir,exist_ok=True)

#         model = GPS(in_dim=in_dim, channels=channels, pe_dim=actual_rwse_dim,
#                     num_layers=args.num_layers, conv_type=args.conv_type, edge_dim=edge_dim).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
#         criterion = nn.BCEWithLogitsLoss() if args.dataset=="ogbg-molhiv" else nn.CrossEntropyLoss()
#         scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0,e/args.warmup_epochs))

#         best_val_metric = -np.inf
#         best_model_path = os.path.join(run_dir, "best_model.pt")

#         for epoch in range(1,args.epochs+1):
#             loss = train_epoch(model, train_loader, optimizer, device, criterion)
#             y_val, y_pred_val = predict(model, valid_loader, device)
#             opt_thresh = compute_optimal_threshold(y_val, y_pred_val)
#             metrics, _ = compute_metrics(y_val, y_pred_val, optimal_threshold=opt_thresh)
#             val_metric = metrics[args.metric]

#             if val_metric > best_val_metric:
#                 best_val_metric = val_metric
#                 torch.save(model.state_dict(), best_model_path)

#             print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val {args.metric} {val_metric:.4f}")
#             scheduler.step()
        
#         # -----------------------------------------------------
#         # START: MODIFIED RESULTS SAVING LOGIC FOR A SINGLE RUN
#         # -----------------------------------------------------
#         # Evaluate best model on the test set
#         print("Evaluating best model on test set...")
#         model.load_state_dict(torch.load(best_model_path))
        
#         # Get predictions for test set
#         y_true_test, y_prob_test = predict(model, test_loader, device)
        
#         # Determine optimal threshold from validation set to avoid data leakage
#         y_true_val, y_prob_val = predict(model, valid_loader, device)
#         opt_thresh = compute_optimal_threshold(y_true_val, y_prob_val)

#         # Compute metrics for test set using both default and optimal thresholds
#         metrics_default, y_pred_def = compute_metrics(y_true_test, y_prob_test)
#         metrics_opt, y_pred_opt = compute_metrics(y_true_test, y_prob_test, optimal_threshold=opt_thresh)
        
#         # Structure the results for this run as per metrics.json
#         run_results = {
#             "best_validation_metric": best_val_metric,
#             "accuracy": metrics_opt['acc'],
#             "balanced_accuracy": metrics_opt['balacc'],
#             "auc_roc": metrics_opt['rocauc'],
#             "auc_pr": metrics_opt['prauc'],
#             "classification_report_default": classification_report(y_true_test, y_pred_def, zero_division=0, output_dict=True),
#             "classification_report_optimal": classification_report(y_true_test, y_pred_opt, zero_division=0, output_dict=True)
#         }

#         # Save the structured results to metrics.json
#         with open(os.path.join(run_dir, "metrics.json"), "w") as f:
#             json.dump(run_results, f, indent=4)
#         print(f"Run {run} results saved at {run_dir}")
#         # ---------------------------------------------------
#         # END: MODIFIED RESULTS SAVING LOGIC FOR A SINGLE RUN
#         # ---------------------------------------------------

#     # -------------------------------------
#     # START: MODIFIED SUMMARY SAVING LOGIC
#     # -------------------------------------
#     print("\nSummarizing results across all runs...")
#     summary_dir = os.path.join(result_dir, "summary")
#     os.makedirs(summary_dir, exist_ok=True)
    
#     all_best_val_metrics = []
#     all_test_metrics = {
#         'accuracy': [],
#         'balanced_accuracy': [],
#         'auc_roc': [],
#         'auc_pr': [],
#     }
#     all_reports_default = []
#     all_reports_optimal = []
    
#     # Collect results from each run's metrics.json
#     for run in range(1, args.runs + 1):
#         run_metrics_path = os.path.join(result_dir, f"run_{run}", "metrics.json")
#         with open(run_metrics_path, "r") as f:
#             run_metrics = json.load(f)
        
#         all_best_val_metrics.append(run_metrics["best_validation_metric"])
        
#         for key in all_test_metrics.keys():
#             all_test_metrics[key].append(run_metrics[key])

#         all_reports_default.append(run_metrics["classification_report_default"])
#         all_reports_optimal.append(run_metrics["classification_report_optimal"])
        
#     # Create and save summary.json with mean and std dev
#     summary_output = {
#         "metric": args.metric,
#         "best_valid_mean": np.mean(all_best_val_metrics),
#         "best_valid_std": np.std(all_best_val_metrics),
#     }
#     for key, values in all_test_metrics.items():
#         summary_output[f"{key}_mean"] = np.mean(values)
#         summary_output[f"{key}_std"] = np.std(values)

#     with open(os.path.join(summary_dir, "summary.json"), "w") as f:
#         json.dump(summary_output, f, indent=4)

#     # Create and save the collected classification reports
#     with open(os.path.join(summary_dir, "reports_default.json"), "w") as f:
#         json.dump(all_reports_default, f, indent=4)
    
#     with open(os.path.join(summary_dir, "reports_optimal.json"), "w") as f:
#         json.dump(all_reports_optimal, f, indent=4)
        
#     print(f"Summary reports saved in {summary_dir}")
#     # -----------------------------------
#     # END: MODIFIED SUMMARY SAVING LOGIC
#     # -----------------------------------








