

# main_graphclass.py
import argparse
import torch
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
from gt_models import Graphormer, GraphiT, GPS 
from training_utils import train, evaluate_graphclass
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
    params = [f"lr-{args.lr}", f"pool-{args.pool}", f"metric-{args.metric}", f"hid-{args.hidden_channels}", f"do-{args.dropout}",
              f"layers-{args.num_layers}"]
    if args.model_family == 'gt' or args.model_name == 'gat':
        params.append(f"heads-{args.nhead}")
    if args.model_family == 'gnn':
        params.append(f"bn-{args.use_bn}")
        params.append(f"res-{args.use_residual}")
    return "_".join(params)

def main():
    parser = argparse.ArgumentParser(description='Unified Pipeline for Graph Classification')
    parser_add_main_args(parser)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(args.seed)
    print(args)

    # --- Data Loading ---
    if args.dataset_type == 'ogb':
        with torch.serialization.safe_globals([DataEdgeAttr]):
            dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')
        # dataset = PygGraphPropPredDataset(name=args.dataset, root='data/OGB')
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
        evaluator = Evaluator(name=args.dataset)
        num_tasks = dataset.num_tasks
        num_features = dataset.num_features
        is_binary = True
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

    # --- Metric Tracking Setup ---
    all_best_valid = [] # MODIFIED: Added list to track best validation scores
    all_acc, all_bal_acc, all_roc_auc, all_pr_auc = [], [], [], []
    all_reports_default, all_reports_optimal = [], []

    for run in range(1, args.runs + 1):
        print(f"\n--- Starting Run {run}/{args.runs} ---")
        
        models = {'gcn': GCN, 'gat': GAT, 'sage': SAGE, 'gin': GIN, 
                  'graphormer': Graphormer, 'graphit': GraphiT, 'gps': GPS}
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
            train(model, train_loader, optimizer, device, args.dataset_type)
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
        out_dir = f'results/{args.dataset}/{args.model_name}/{param_string}/run_{run}'
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
    summary_dir = f'results/{args.dataset}/{args.model_name}/{param_string}/summary'
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