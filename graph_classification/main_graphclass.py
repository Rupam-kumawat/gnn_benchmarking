


# main_graphclass.py

import argparse
import torch
import numpy as np
import os
import json
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score, balanced_accuracy_score

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# --- Model Imports ---
from parse_graphclass import parser_add_main_args
from gnn_models import GCN, GAT, SAGE, GIN
# from gps_model import GPSModel
from subgraphormer_model import Subgraphormer
# from graphvit_model import GraphViT

# --- Utility Imports ---
from training_utils import train, evaluate_graphclass
from pe_layer import compute_laplacian_pe
from subgraphormer_utils import get_subgraphormer_transform
# from graphvit_utils import get_graphvit_transform

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


def fix_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_param_string(args):
    """Generates a unique, readable string from all relevant hyperparameters."""
    params = [f"lr-{args.lr}", f"hid-{args.hidden_channels}", f"do-{args.dropout}",
              f"layers-{args.num_layers}", f"pool-{args.pool}", f"metric-{args.metric}"]
    if args.model_name in ['gat', 'gps', 'graphvit', 'subgraphormer']:
        params.append(f"heads-{args.nhead}")
    if args.model_name == 'gps':
        params.append(f"local-{args.local_gnn}")
        if args.use_lap_pe:
            params.append(f"lapPE-{args.lap_pe_dim}")
    return "_".join(params)

def main():
    parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification')
    parser_add_main_args(parser)
    args = parser.parse_args

    print('args', args)
    exit()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(args.seed)
    print(args)
    
    # --- Data Loading & Transformation ---
    transform = None
    if args.model_name == 'subgraphormer':
        print("Using Subgraphormer transform...")
        transform = get_subgraphormer_transform(args)
    elif args.model_name == 'graphvit':
        print("Using Graph-ViT transform...")
        transform = get_graphvit_transform(args)

    if args.dataset_type == 'ogb':
        with torch.serialization.safe_globals([DataEdgeAttr]):
            dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB', transform=transform)
            
        # dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB', transform=transform)
        if args.use_lap_pe:
            print("Pre-computing Laplacian PE for OGB dataset...")
            for g in tqdm(dataset, desc="Computing LapPE"):
                compute_laplacian_pe(g, args.lap_pe_dim)
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
        evaluator = Evaluator(name=args.dataset)
        num_tasks = dataset.num_tasks
        num_features = dataset.num_features
        is_binary = True
    else: # TU Datasets
        dataset = TUDataset(root='data/TUDataset', name=args.dataset, transform=transform)
        if dataset.num_node_features == 0 and not transform: # Avoid applying OneHotDegree if a custom transform exists
            max_deg = 0
            for g in dataset:
                if hasattr(g, 'edge_index') and g.edge_index is not None and g.edge_index.numel() > 0:
                    max_deg = max(max_deg, int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()))
            dataset.transform = OneHotDegree(max_degree=max_deg)
        
        if args.use_lap_pe and not transform:
            print("Pre-computing Laplacian PE for TU dataset...")
            for g in tqdm(dataset, desc="Computing LapPE"):
                compute_laplacian_pe(g, args.lap_pe_dim)
        
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

    # --- Metric Tracking and Training Loop ---
    all_best_valid, all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], [], []
    all_reports_default, all_reports_optimal = [], []

    for run in range(1, args.runs + 1):
        print(f"\n--- Starting Run {run}/{args.runs} ---")
        
        use_ogb = True if args.dataset_type == 'ogb' else False
        
        # --- Model Initialization ---
        models = {
            'gcn': GCN, 'gat': GAT, 'sage': SAGE, 'gin': GIN,
            'gps': GPSModel, 'subgraphormer': Subgraphormer, 'graphvit': GraphViT
        }
        ModelClass = models.get(args.model_name)

        if ModelClass is None:
            raise ValueError(f"Model {args.model_name} not recognized.")
        
        # Instantiate the correct model with its specific arguments
        if args.model_name in ['gcn', 'gat', 'sage', 'gin']:
            model = ModelClass(
                num_features, num_tasks, 
                hidden=args.hidden_channels, num_layers=args.num_layers, 
                dropout=args.dropout, pool=args.pool, 
                use_ogb_features=use_ogb, use_bn=args.use_bn, use_residual=args.use_residual
            ).to(device)
        elif args.model_name == 'gps':
            model = GPSModel(
                num_features=num_features, num_classes=num_tasks,
                hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                nhead=args.nhead, dropout=args.dropout, pool=args.pool,
                local_gnn_type=args.local_gnn, global_model_type=args.global_model,
                use_ogb_features=use_ogb, use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim
            ).to(device)
        elif args.model_name == 'subgraphormer':
            model = Subgraphormer(num_features, num_tasks, args, dataset_type=args.dataset_type).to(device)
        elif args.model_name == 'graphvit':
            model = GraphViT(num_features, num_tasks, args).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_valid_metric, best_model_state = -1, None

        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, optimizer, device, args.dataset_type)
            _, _, valid_perf = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
            print(f"Epoch {epoch:02d} | Valid Performance: {valid_perf}")
            
            if valid_perf.get(args.metric, -1) > best_valid_metric:
                best_valid_metric = valid_perf[args.metric]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        print(f"Run {run} finished. Best validation {args.metric}: {best_valid_metric:.4f}")
        all_best_valid.append(best_valid_metric)
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # --- Final Evaluation ---
        y_valid_true, y_valid_pred, _ = evaluate_graphclass(model, valid_loader, device, args.dataset_type, evaluator)
        y_test_true, y_test_pred, test_metrics_default = evaluate_graphclass(model, test_loader, device, args.dataset_type, evaluator)
        
        # ... (rest of evaluation, reporting, and saving logic is the same) ...
        report_optimal = None
        if is_binary and y_valid_pred is not None and y_valid_pred.numel() > 0:
            if y_valid_pred.shape[1] == 1:
                valid_probs = torch.sigmoid(y_valid_pred).squeeze().cpu().numpy()
            else:
                valid_probs = torch.softmax(y_valid_pred, dim=1)[:, 1].cpu().numpy()
            
            valid_true = y_valid_true.cpu().numpy().squeeze()
            best_f1, best_threshold = -1, 0.5
            for threshold in np.linspace(0, 1, 100):
                f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro', zero_division=0)
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

        # Aggregating and Saving Run Results
        acc = test_metrics_default.get('acc', 0)
        bal_acc = test_metrics_default.get('balacc', 0)
        roc_auc = test_metrics_default.get('rocauc', 0)
        pr_auc = test_metrics_default.get('prauc', 0)
        
        y_pred_default_np = torch.argmax(y_test_pred, dim=1).cpu().numpy() if y_test_pred.shape[1] > 1 else (y_test_pred.cpu().numpy() > 0).astype(int)
        y_test_true_np = y_test_true.cpu().numpy()
        report_default = classification_report(y_test_true_np.squeeze(), y_pred_default_np.squeeze(), output_dict=True, zero_division=0)
        
        all_acc.append(acc); all_bal_acc.append(bal_acc); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc)
        all_reports_default.append(report_default)
        if report_optimal: all_reports_optimal.append(report_optimal)

        param_string = generate_param_string(args)
        out_dir = f'results/{args.dataset}/{args.model_name}/{param_string}/run_{run}'
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
        
        run_metrics = {'best_validation_metric': best_valid_metric, 'accuracy': acc, 'balanced_accuracy': bal_acc, 
                       'auc_roc': roc_auc, 'auc_pr': pr_auc,
                       'classification_report_default': report_default, 
                       'classification_report_optimal': report_optimal}
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(run_metrics, f, indent=2)

    # --- Final Summary ---
    param_string = generate_param_string(args)
    summary_dir = f'results/{args.dataset}/{args.model_name}/{param_string}/summary'
    os.makedirs(summary_dir, exist_ok=True)
    
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