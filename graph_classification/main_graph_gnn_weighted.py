import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from parse_graphclass import parser_add_main_args
from gnn_models import GCN, GAT, SAGE, GIN
# from ignore_gt_models import Graphormer, GraphiT, GPS
# from training_utils import train, evaluate_graphclass # MODIFIED: We will define train locally
from training_utils import evaluate_graphclass # MODIFIED: Only import evaluate
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# =====================================================================================
#  ADDED: Loss Functions
# =====================================================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss, compatible with `BCEWithLogitsLoss` (expects raw logits).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
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
            return loss

class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss, compatible with `NLLLoss` (expects log-probabilities).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # [C] tensor of weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets):
        log_pt = -F.nll_loss(log_probs, targets, reduction='none')
        pt = log_pt.exp() # [N]
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -log_pt * focal_term
        
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha).to(log_probs.device)
            alpha_t = self.alpha.gather(0, targets) # [N]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# =====================================================================================
#  ADDED: Local Train Function
# =====================================================================================

def train(model, loader, optimizer, device, criterion, dataset_type, num_tasks):
    """
    MODIFIED: Local train function to handle custom criterion.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data) # Output shape [B, num_tasks]

        if dataset_type == 'ogb':
            # OGB graph prop (e.g., molhiv) expects logits for BCEWithLogitsLoss.
            # Labels are shape [B, num_tasks], float.
            y_true = data.y.to(torch.float)
            loss = criterion(out, y_true)
        else:
            # TUDatasets expect log_softmax for NLLLoss.
            # Labels are shape [B] or [B, 1], long.
            y_true = data.y.squeeze().to(torch.long)
            out_log_softmax = F.log_softmax(out, dim=1)
            loss = criterion(out_log_softmax, y_true)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    # Return average loss
    return total_loss / len(loader.dataset)

# =====================================================================================

def fix_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_param_string(args):
    """Generates a unique, readable string from all relevant hyperparameters."""
    params = [f"lr-{args.lr}", f"pool-{args.pool}", f"metric-{args.metric}", f"hid-{args.hidden_channels}", f"do-{args.dropout}",
              f"layers-{args.num_layers}"]
    if args.model_family == 'gt' or args.model_name == 'gat':
        params.append(f"heads-{args.nhead}")
    if args.model_family == 'gnn':
        params.append(f"bn-{args.use_bn}")
        params.append(f"res-{args.use_residual}")
        
    # --- MODIFIED: Add loss flags to param string ---
    if args.use_class_weight: params.append('class_weight')
    if args.use_focal_loss: params.append(f'focal_g-{args.focal_gamma}')
    # Only add focal_alpha if it's a binary task (OGB)
    if args.use_focal_loss and args.dataset_type == 'ogb': 
        params.append(f'focal_a-{args.focal_alpha}')
    # --- End Modification ---
        
    return "_".join(params)

def main():
    parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification')
    parser_add_main_args(parser)
    
    # --- MODIFIED: Add arguments for weighted loss ---
    parser.add_argument('--use_class_weight', action='store_true',
                        help='Use inverse class frequency to weight the loss function.')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of standard NLL/BCE Loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma focusing parameter for Focal Loss.')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha balancing parameter for Binary Focal Loss (for OGB).')
    # --- End Modification ---

    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(args.seed)
    print(args)

    # --- Data Loading ---
    if args.dataset_type == 'ogb':

        if hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals([DataEdgeAttr]):
                dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')#, transform=transform)
        else:
            # Fallback for older PyTorch versions where safe_globals doesn't exist
            dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')#, transform=transform)
            
        # with torch.serialization.safe_globals([DataEdgeAttr]):
        #     dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')
        # dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
        evaluator = Evaluator(name=args.dataset)
        num_tasks = dataset.num_tasks
        num_features = dataset.num_features
        is_binary = True # OGB graph prop tasks are binary/multilabel
    else: # TU Datasets
        dataset = TUDataset(root='data/TUDataset', name=args.dataset)
        if dataset.num_node_features == 0:
            max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset if g.edge_index is not None and g.edge_index.numel() > 0)
            dataset.transform = OneHotDegree(max_degree=max_deg)
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        test_size = len(dataset) // 10
        valid_size = len(dataset) // 10
        train_size = len(dataset) - test_size - valid_size
        
        test_dataset = dataset[indices[:test_size]]
        valid_dataset = dataset[indices[test_size:test_size + valid_size]]
        train_dataset = dataset[indices[test_size + valid_size:]]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        evaluator = None
        num_tasks = dataset.num_classes
        num_features = dataset.num_features
        print('num_features', num_features)
        is_binary = dataset.num_classes == 2

    # --- MODIFIED: Loss Function Setup ---
    class_weights, pos_weight = None, None
    if args.use_class_weight:
        print("--- Calculating Class Weights ---")
        # Get all labels from the full dataset (before splitting)
        y_all = dataset.y
        if y_all is None: # TUDataset stores it differently
            y_all = torch.cat([data.y for data in dataset], dim=0)
        
        if args.dataset_type == 'ogb': # Binary/Multilabel BCE
            y = y_all.float()
            pos_samples = y.sum(dim=0)
            total_samples = len(y)
            neg_samples = total_samples - pos_samples
            pos_weight = neg_samples / (pos_samples + 1e-8)
            pos_weight = pos_weight.to(device)
            print(f"OGB problem. pos_weight = {pos_weight}")
        else: # Multiclass NLL
            y = y_all.squeeze().to(torch.long)
            class_counts = torch.bincount(y)
            # Use inverse frequency: total / (n_classes * count)
            class_weights = y.shape[0] / (num_tasks * class_counts.float())
            class_weights[torch.isinf(class_weights)] = 0.0
            class_weights = class_weights.to(device)
            print(f"TUDataset problem. Weights: {class_weights}")

    # --- Select Criterion ---
    if args.dataset_type == 'ogb': 
        if args.use_focal_loss:
            print(f"Using BinaryFocalLoss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
            criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        else:
            print(f"Using BCEWithLogitsLoss (pos_weight={pos_weight is not None})")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else: # TUDataset
        if args.use_focal_loss:
            print(f"Using FocalLoss (gamma={args.focal_gamma}, alpha={'class_weights' if class_weights is not None else 'None'})")
            criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        else:
            print(f"Using NLLLoss (weight={class_weights is not None})")
            criterion = nn.NLLLoss(weight=class_weights)
    # --- END MODIFIED SECTION ---

    # --- Metric Tracking Setup ---
    all_best_valid = [] # MODIFIED: Added list to track best validation scores
    all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
    all_reports_default, all_reports_optimal = [], []

    for run in range(1, args.runs + 1):
        print(f"\n--- Starting Run {run}/{args.runs} ---")
        
        models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE}
        # , 'gin': GIN, 
        #           'graphormer': Graphormer, 'graphit': GraphiT, 'gps': GPS}
        ModelClass = models[args.model_name]
        use_ogb = True if args.dataset_type == 'ogb' else False
        
        if args.model_family == 'gnn':
            model = ModelClass(num_features, num_tasks, hidden=args.hidden_channels, num_layers=args.num_layers, 
                               dropout=args.dropout, pool=args.pool, use_ogb_features=use_ogb, use_bn=args.use_bn, 
                               use_residual=args.use_residual).to(device)
        else: # Graph Transformer
            model = ModelClass(num_features, num_tasks, num_layer=args.num_layers, hidden_channels=args.hidden_channels, 
                               nhead=args.nhead, dropout=args.dropout, pool=args.pool, use_ogb_features=use_ogb).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_valid_metric, best_model_state = -1, None

        for epoch in range(1, args.epochs + 1):
            # --- MODIFIED: Call local train function with criterion ---
            train(model, train_loader, optimizer, device, criterion, args.dataset_type, num_tasks)
            # --- End Modification ---
            
            _, _, valid_perf = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
            
            print(f"Epoch {epoch:02d} | Valid Performance: {valid_perf}")
            
            if valid_perf[args.metric] > best_valid_metric:
                best_valid_metric = valid_perf[args.metric]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        print(f"Run {run} finished. Best validation {args.metric}: {best_valid_metric:.4f}")
        all_best_valid.append(best_valid_metric) # MODIFIED: Record the best validation score for this run
        model.load_state_dict(best_model_state)

        # --- Final Evaluation with Optimal Thresholding ---
        y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
        y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

        report_optimal = None
        
        if is_binary:
            if y_valid_pred.shape[1] == 1:
                valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy()
            else:
                valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
            
            valid_true = y_valid_true.cpu().numpy().squeeze()
            best_f1 = -1
            best_threshold = 0.5
            for threshold in np.linspace(0, 1, 100):
                f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
                if f1 > best_f1:
                    best_f1, best_threshold = f1, threshold
            
            if y_test_pred.shape[1] == 1:
                test_probs_np = torch.sigmoid(y_test_pred).squeeze().cpu().numpy()
            else:
                test_probs_np = torch.softmax(y_test_pred, dim=1)[:, 1].cpu().numpy()

            y_test_true_np = y_test_true.cpu().numpy().squeeze()
            y_pred_optimal_np = (test_probs_np >= best_threshold).astype(int)
            
            test_acc_opt = accuracy_score(y_test_true_np, y_pred_optimal_np)
            test_bal_acc_opt = balanced_accuracy_score(y_test_true_np, y_pred_optimal_np)
            report_optimal = classification_report(y_test_true_np, y_pred_optimal_np, output_dict=True, zero_division=0)
            
            test_metrics_default['acc'] = test_acc_opt
            test_metrics_default['balacc'] = test_bal_acc_opt

        # --- Aggregating and Saving Run Results ---
        acc = test_metrics_default['acc']
        bal_acc = test_metrics_default['balacc']
        roc_auc = test_metrics_default['rocauc']
        pr_auc = test_metrics_default.get('prauc', test_metrics_default.get('ap', 0))
        
        y_pred_default_np = torch.argmax(y_test_pred, dim=1).cpu().numpy() if y_test_pred.shape[1] > 1 else (y_test_pred.cpu().numpy() > 0).astype(int)
        y_test_true_np = y_test_true.cpu().numpy()
        report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
        
        all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
        all_reports_default.append(report_default)
        if report_optimal: all_reports_optimal.append(report_optimal)

        param_string = generate_param_string(args)
        # --- MODIFIED: Changed 'results' to 'results_weighted' ---
        out_dir = f'results_weighted/{args.dataset}/{args.model_name}/{param_string}/run_{run}'
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
        
        # MODIFIED: Add best_valid_metric to the per-run JSON
        run_metrics = {'best_validation_metric': best_valid_metric, 'accuracy': acc, 'balanced_accuracy': bal_acc, 
                         'auc_roc': roc_auc, 'auc_pr': pr_auc,
                         'classification_report_default': report_default, 
                         'classification_report_optimal': report_optimal}
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(run_metrics, f, indent=2)

    # --- Final Summary ---
    param_string = generate_param_string(args)
    # --- MODIFIED: Changed 'results' to 'results_weighted' ---
    summary_dir = f'results_weighted/{args.dataset}/{args.model_name}/{param_string}/summary'
    os.makedirs(summary_dir, exist_ok=True)
    
    # MODIFIED: Add validation metric to the final summary
    summary = {
        'metric': args.metric,
        'best_valid_mean': np.mean(all_best_valid), 'best_valid_std': np.std(all_best_valid),
        'accuracy_mean': np.mean(all_acc), 'accuracy_std': np.std(all_acc),
        'balanced_accuracy_mean': np.mean(all_bal_acc), 'balanced_accuracy_std': np.std(all_bal_acc),
        'auc_roc_mean': np.mean(all_roc_auc), 'auc_roc_std': np.std(all_roc_auc),
        'auc_pr_mean': np.mean(all_pr_auc), 'auc_pr_std': np.std(all_pr_auc),
    }
    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(summary_dir, 'reports_default.json'), 'w') as f:
        json.dump(all_reports_default, f, indent=2)
    if all_reports_optimal:
        with open(os.path.join(summary_dir, 'reports_optimal.json'), 'w') as f:
            json.dump(all_reports_optimal, f, indent=2)
    
    print(f"\n{'='*40}\nFINAL SUMMARY ({args.metric})\n{'='*40}")
    print(json.dumps(summary, indent=2))

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
# from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score

# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
# from torch_geometric.transforms import OneHotDegree
# from torch_geometric.utils import degree
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# from parse_graphclass import parser_add_main_args
# from gnn_models import GCN, GAT, SAGE, GIN
# # from ignore_gt_models import Graphormer, GraphiT, GPS
# # from training_utils import train, evaluate_graphclass # MODIFIED: We will define train locally
# from training_utils import evaluate_graphclass # MODIFIED: Only import evaluate
# # from torch_geometric.data.storage import GlobalStorage
# # from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# # torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# # =====================================================================================
# #  ADDED: Loss Functions
# # =====================================================================================

# class BinaryFocalLoss(nn.Module):
#     """
#     Binary Focal Loss, compatible with `BCEWithLogitsLoss` (expects raw logits).
#     """
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(BinaryFocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, logits, targets):
#         loss_bce = self.bce_loss(logits, targets)
#         pt = torch.exp(-loss_bce)
#         focal_term = (1.0 - pt).pow(self.gamma)
#         loss = focal_term * loss_bce
        
#         if self.alpha is not None:
#             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#             loss = alpha_t * loss

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

# class FocalLoss(nn.Module):
#     """
#     Multiclass Focal Loss, compatible with `NLLLoss` (expects log-probabilities).
#     """
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha # [C] tensor of weights
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, log_probs, targets):
#         log_pt = -F.nll_loss(log_probs, targets, reduction='none')
#         pt = log_pt.exp() # [N]
#         focal_term = (1.0 - pt).pow(self.gamma)
#         loss = -log_pt * focal_term
        
#         if self.alpha is not None:
#             if not isinstance(self.alpha, torch.Tensor):
#                 self.alpha = torch.tensor(self.alpha).to(log_probs.device)
#             alpha_t = self.alpha.gather(0, targets) # [N]
#             loss = alpha_t * loss

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

# # =====================================================================================
# #  ADDED: Local Train Function
# # =====================================================================================

# def train(model, loader, optimizer, device, criterion, dataset_type, num_tasks):
#     """
#     MODIFIED: Local train function to handle custom criterion.
#     """
#     model.train()
#     total_loss = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data) # Output shape [B, num_tasks]

#         if dataset_type == 'ogb':
#             # OGB graph prop (e.g., molhiv) expects logits for BCEWithLogitsLoss.
#             # Labels are shape [B, num_tasks], float.
#             y_true = data.y.to(torch.float)
#             loss = criterion(out, y_true)
#         else:
#             # TUDatasets expect log_softmax for NLLLoss.
#             # Labels are shape [B] or [B, 1], long.
#             y_true = data.y.squeeze().to(torch.long)
#             out_log_softmax = F.log_softmax(out, dim=1)
#             loss = criterion(out_log_softmax, y_true)

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs
#     # Return average loss
#     return total_loss / len(loader.dataset)

# # =====================================================================================

# def fix_seed(seed):
#     """Sets the seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def generate_param_string(args):
#     """Generates a unique, readable string from all relevant hyperparameters."""
#     params = [f"lr-{args.lr}", f"pool-{args.pool}", f"metric-{args.metric}", f"hid-{args.hidden_channels}", f"do-{args.dropout}",
#               f"layers-{args.num_layers}"]
#     if args.model_family == 'gt' or args.model_name == 'gat':
#         params.append(f"heads-{args.nhead}")
#     if args.model_family == 'gnn':
#         params.append(f"bn-{args.use_bn}")
#         params.append(f"res-{args.use_residual}")
        
#     # --- MODIFIED: Add loss flags to param string ---
#     if args.use_class_weight: params.append('class_weight')
#     if args.use_focal_loss: params.append(f'focal_g-{args.focal_gamma}')
#     # Only add focal_alpha if it's a binary task (OGB)
#     if args.use_focal_loss and args.dataset_type == 'ogb': 
#         params.append(f'focal_a-{args.focal_alpha}')
#     # --- End Modification ---
        
#     return "_".join(params)

# def main():
#     parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification')
#     parser_add_main_args(parser)
    
#     # --- MODIFIED: Add arguments for weighted loss ---
#     parser.add_argument('--use_class_weight', action='store_true',
#                         help='Use inverse class frequency to weight the loss function.')
#     parser.add_argument('--use_focal_loss', action='store_true',
#                         help='Use Focal Loss instead of standard NLL/BCE Loss.')
#     parser.add_argument('--focal_gamma', type=float, default=2.0,
#                         help='Gamma focusing parameter for Focal Loss.')
#     parser.add_argument('--focal_alpha', type=float, default=0.25,
#                         help='Alpha balancing parameter for Binary Focal Loss (for OGB).')
#     # --- End Modification ---

#     args = parser.parse_args()
    
#     device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
#     fix_seed(args.seed)
#     print(args)

#     # --- Data Loading ---
#     if args.dataset_type == 'ogb':
#         # with torch.serialization.safe_globals([DataEdgeAttr]):
#         dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')
#         # dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')
#         split_idx = dataset.get_idx_split()
#         train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
#         valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
#         test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
#         evaluator = Evaluator(name=args.dataset)
#         num_tasks = dataset.num_tasks
#         num_features = dataset.num_features
#         is_binary = True # OGB graph prop tasks are binary/multilabel
#     else: # TU Datasets
#         dataset = TUDataset(root='data/TUDataset', name=args.dataset)
#         if dataset.num_node_features == 0:
#             max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset if g.edge_index is not None and g.edge_index.numel() > 0)
#             dataset.transform = OneHotDegree(max_degree=max_deg)
        
#         indices = list(range(len(dataset)))
#         random.shuffle(indices)
#         test_size = len(dataset) // 10
#         valid_size = len(dataset) // 10
#         train_size = len(dataset) - test_size - valid_size
        
#         test_dataset = dataset[indices[:test_size]]
#         valid_dataset = dataset[indices[test_size:test_size + valid_size]]
#         train_dataset = dataset[indices[test_size + valid_size:]]

#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#         valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
#         test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
#         evaluator = None
#         num_tasks = dataset.num_classes
#         num_features = dataset.num_features
#         print('num_features', num_features)
#         is_binary = dataset.num_classes == 2

#     # --- MODIFIED: Loss Function Setup ---
#     class_weights, pos_weight = None, None
#     if args.use_class_weight:
#         print("--- Calculating Class Weights ---")
#         # Get all labels from the full dataset (before splitting)
#         y_all = dataset.y
#         if y_all is None: # TUDataset stores it differently
#             y_all = torch.cat([data.y for data in dataset], dim=0)
        
#         if args.dataset_type == 'ogb': # Binary/Multilabel BCE
#             y = y_all.float()
#             pos_samples = y.sum(dim=0)
#             total_samples = len(y)
#             neg_samples = total_samples - pos_samples
#             pos_weight = neg_samples / (pos_samples + 1e-8)
#             pos_weight = pos_weight.to(device)
#             print(f"OGB problem. pos_weight = {pos_weight}")
#         else: # Multiclass NLL
#             y = y_all.squeeze().to(torch.long)
#             class_counts = torch.bincount(y)
#             # Use inverse frequency: total / (n_classes * count)
#             class_weights = y.shape[0] / (num_tasks * class_counts.float())
#             class_weights[torch.isinf(class_weights)] = 0.0
#             class_weights = class_weights.to(device)
#             print(f"TUDataset problem. Weights: {class_weights}")

#     # --- Select Criterion ---
#     if args.dataset_type == 'ogb': 
#         if args.use_focal_loss:
#             print(f"Using BinaryFocalLoss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
#             criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
#         else:
#             print(f"Using BCEWithLogitsLoss (pos_weight={pos_weight is not None})")
#             criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     else: # TUDataset
#         if args.use_focal_loss:
#             print(f"Using FocalLoss (gamma={args.focal_gamma}, alpha={'class_weights' if class_weights is not None else 'None'})")
#             criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
#         else:
#             print(f"Using NLLLoss (weight={class_weights is not None})")
#             criterion = nn.NLLLoss(weight=class_weights)
#     # --- END MODIFIED SECTION ---

#     # --- Metric Tracking Setup ---
#     all_best_valid = [] # MODIFIED: Added list to track best validation scores
#     all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
#     all_reports_default, all_reports_optimal = [], []

#     for run in range(1, args.runs + 1):
#         print(f"\n--- Starting Run {run}/{args.runs} ---")
        
#         models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE}
#         # , 'gin': GIN, 
#         #           'graphormer': Graphormer, 'graphit': GraphiT, 'gps': GPS}
#         ModelClass = models[args.model_name]
#         use_ogb = True if args.dataset_type == 'ogb' else False
        
#         if args.model_family == 'gnn':
#             model = ModelClass(num_features, num_tasks, hidden=args.hidden_channels, num_layers=args.num_layers, 
#                                dropout=args.dropout, pool=args.pool, use_ogb_features=use_ogb, use_bn=args.use_bn, 
#                                use_residual=args.use_residual).to(device)
#         else: # Graph Transformer
#             model = ModelClass(num_features, num_tasks, num_layer=args.num_layers, hidden_channels=args.hidden_channels, 
#                                nhead=args.nhead, dropout=args.dropout, pool=args.pool, use_ogb_features=use_ogb).to(device)

#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#         best_valid_metric, best_model_state = -1, None

#         for epoch in range(1, args.epochs + 1):
#             # --- MODIFIED: Call local train function with criterion ---
#             train(model, train_loader, optimizer, device, criterion, args.dataset_type, num_tasks)
#             # --- End Modification ---
            
#             _, _, valid_perf = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
            
#             print(f"Epoch {epoch:02d} | Valid Performance: {valid_perf}")
            
#             if valid_perf[args.metric] > best_valid_metric:
#                 best_valid_metric = valid_perf[args.metric]
#                 best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
#         print(f"Run {run} finished. Best validation {args.metric}: {best_valid_metric:.4f}")
#         all_best_valid.append(best_valid_metric) # MODIFIED: Record the best validation score for this run
#         model.load_state_dict(best_model_state)

#         # --- Final Evaluation with Optimal Thresholding ---
#         y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
#         y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)

#         report_optimal = None
        
#         if is_binary:
#             if y_valid_pred.shape[1] == 1:
#                 valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy()
#             else:
#                 valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
            
#             valid_true = y_valid_true.cpu().numpy().squeeze()
#             best_f1 = -1
#             best_threshold = 0.5
#             for threshold in np.linspace(0, 1, 100):
#                 f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
#                 if f1 > best_f1:
#                     best_f1, best_threshold = f1, threshold
            
#             if y_test_pred.shape[1] == 1:
#                 test_probs_np = torch.sigmoid(y_test_pred).squeeze().cpu().numpy()
#             else:
#                 test_probs_np = torch.softmax(y_test_pred, dim=1)[:, 1].cpu().numpy()

#             y_test_true_np = y_test_true.cpu().numpy().squeeze()
#             y_pred_optimal_np = (test_probs_np >= best_threshold).astype(int)
            
#             test_acc_opt = accuracy_score(y_test_true_np, y_pred_optimal_np)
#             test_bal_acc_opt = balanced_accuracy_score(y_test_true_np, y_pred_optimal_np)
#             report_optimal = classification_report(y_test_true_np, y_pred_optimal_np, output_dict=True, zero_division=0)
            
#             test_metrics_default['acc'] = test_acc_opt
#             test_metrics_default['balacc'] = test_bal_acc_opt

#         # --- Aggregating and Saving Run Results ---
#         acc = test_metrics_default['acc']
#         bal_acc = test_metrics_default['balacc']
#         roc_auc = test_metrics_default['rocauc']
#         pr_auc = test_metrics_default.get('prauc', test_metrics_default.get('ap', 0))
        
#         y_pred_default_np = torch.argmax(y_test_pred, dim=1).cpu().numpy() if y_test_pred.shape[1] > 1 else (y_test_pred.cpu().numpy() > 0).astype(int)
#         y_test_true_np = y_test_true.cpu().numpy()
#         report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
        
#         all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
#         all_reports_default.append(report_default)
#         if report_optimal: all_reports_optimal.append(report_optimal)

#         param_string = generate_param_string(args)
#         # --- MODIFIED: Changed 'results' to 'results_weighted' ---
#         out_dir = f'results_weighted/{args.dataset}/{args.model_name}/{param_string}/run_{run}'
#         os.makedirs(out_dir, exist_ok=True)
#         torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
        
#         # MODIFIED: Add best_valid_metric to the per-run JSON
#         run_metrics = {'best_validation_metric': best_valid_metric, 'accuracy': acc, 'balanced_accuracy': bal_acc, 
#                          'auc_roc': roc_auc, 'auc_pr': pr_auc,
#                          'classification_report_default': report_default, 
#                          'classification_report_optimal': report_optimal}
#         with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
#             json.dump(run_metrics, f, indent=2)

#     # --- Final Summary ---
#     param_string = generate_param_string(args)
#     # --- MODIFIED: Changed 'results' to 'results_weighted' ---
#     summary_dir = f'results_weighted/{args.dataset}/{args.model_name}/{param_string}/summary'
#     os.makedirs(summary_dir, exist_ok=True)
    
#     # MODIFIED: Add validation metric to the final summary
#     summary = {
#         'metric': args.metric,
#         'best_valid_mean': np.mean(all_best_valid), 'best_valid_std': np.std(all_best_valid),
#         'accuracy_mean': np.mean(all_acc), 'accuracy_std': np.std(all_acc),
#         'balanced_accuracy_mean': np.mean(all_bal_acc), 'balanced_accuracy_std': np.std(all_bal_acc),
#         'auc_roc_mean': np.mean(all_roc_auc), 'auc_roc_std': np.std(all_roc_auc),
#         'auc_pr_mean': np.mean(all_pr_auc), 'auc_pr_std': np.std(all_pr_auc),
#     }
#     with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)
#     with open(os.path.join(summary_dir, 'reports_default.json'), 'w') as f:
#         json.dump(all_reports_default, f, indent=2)
#     if all_reports_optimal:
#         with open(os.path.join(summary_dir, 'reports_optimal.json'), 'w') as f:
#             json.dump(all_reports_optimal, f, indent=2)
    
#     print(f"\n{'='*40}\nFINAL SUMMARY ({args.metric})\n{'='*40}")
#     print(json.dumps(summary, indent=2))

# if __name__ == "__main__":
#     main()