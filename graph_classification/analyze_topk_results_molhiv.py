import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import random
from sklearn.metrics import ndcg_score
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# --- Import Models & Utils ---
from gnn_models import GCN, GAT, SAGE, GIN
from training_utils import evaluate_graphclass 
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# Fix for PyTorch > 2.6 safe globals
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# =====================================================================================
#   Configuration: Fine-grained then 5% steps
# =====================================================================================
K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))

# =====================================================================================
#   Ranking Metrics Helper
# =====================================================================================

def calculate_ranking_metrics(y_true, y_score, k_percentages):
    y_true = np.asarray(y_true).squeeze()
    y_score = np.asarray(y_score).squeeze()
    
    # Sort by score descending
    sort_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sort_indices]
    
    total_positives = np.sum(y_true)
    total_samples = len(y_true_sorted)
    metrics = {}
    
    for k_pct in k_percentages:
        k = int(np.ceil(total_samples * (k_pct / 100.0)))
        k = max(1, min(k, total_samples)) 
            
        top_k_labels = y_true_sorted[:k]
        num_positives_in_top_k = np.sum(top_k_labels)
        
        metric_key = f"at_{k_pct}_pct"

        metrics[f'precision_{metric_key}'] = float(num_positives_in_top_k / k)
        if total_positives > 0:
            metrics[f'recall_{metric_key}'] = float(num_positives_in_top_k / total_positives)
        else:
            metrics[f'recall_{metric_key}'] = 0.0
        
        true_relevance = y_true.reshape(1, -1)
        pred_scores = y_score.reshape(1, -1)
        metrics[f'ndcg_{metric_key}'] = float(ndcg_score(true_relevance, pred_scores, k=k))

    return metrics

def parse_param_string_to_args(param_string, base_args):
    args = argparse.Namespace(**vars(base_args))
    parts = param_string.split('_')
    key_map = {
        'lr': 'lr', 'pool': 'pool', 'metric': 'metric', 'hid': 'hidden_channels',
        'do': 'dropout', 'layers': 'num_layers', 'heads': 'nhead', 'bn': 'use_bn',
        'res': 'use_residual'
    }
    type_map = {
        'lr': float, 'pool': str, 'metric': str, 'hid': int, 'do': float,
        'layers': int, 'heads': int, 'bn': lambda x: x.lower() == 'true',
        'res': lambda x: x.lower() == 'true'
    }
    for part in parts:
        if '-' not in part: continue
        try:
            key_short, val = part.split('-', 1)
            if key_short in key_map:
                setattr(args, key_map[key_short], type_map[key_short](val))
        except: pass
    return args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================================================
#   Main Execution
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description='Granular Top-K Evaluation')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--base_results_dir', type=str, default='results_weighted')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)

    base_args = parser.parse_args()
    base_args.dataset_type = 'ogb' if 'ogb' in base_args.dataset else 'tu'
    device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(base_args.seed)

    # 1. Load Data
    print(f"--- Loading {base_args.dataset} ---")
    if base_args.dataset_type == 'ogb':
        dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
        split_idx = dataset.get_idx_split()
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=base_args.batch_size, shuffle=False)
        evaluator = Evaluator(name=base_args.dataset)
        num_tasks, num_features = dataset.num_tasks, dataset.num_features
    else:
        dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
        if dataset.num_node_features == 0:
            max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset)
            dataset.transform = OneHotDegree(max_degree=max_deg)
        indices = list(range(len(dataset)))
        random.seed(base_args.seed)
        random.shuffle(indices)
        test_dataset = dataset[indices[:len(dataset)//10]]
        test_loader = DataLoader(test_dataset, batch_size=base_args.batch_size, shuffle=False)
        evaluator = None
        num_tasks, num_features = dataset.num_classes, dataset.num_features

    # 2. Process
    base_path = os.path.join(base_args.base_results_dir, base_args.dataset, base_args.model_name)
    if not os.path.exists(base_path): return

    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) 
            and d not in ['summary', 'topk_results']]
    if base_args.filter: dirs = [d for d in dirs if base_args.filter in d]

    for folder in dirs:
        print(f"Processing: {folder}")
        exp_path = os.path.join(base_path, folder)
        out_dir = os.path.join(exp_path, 'topk_results')
        os.makedirs(out_dir, exist_ok=True)
        args = parse_param_string_to_args(folder, base_args)
        all_rank_metrics = {f'{m}_at_{k}_pct': [] for k in K_PERCENTAGES for m in ['precision', 'recall', 'ndcg']}

        for run in range(1, args.runs + 1):
            run_dir = os.path.join(exp_path, f'run_{run}')
            model_path = os.path.join(run_dir, 'model.pt')
            if not os.path.exists(model_path): continue

            try:
                ModelClass = {'gcn': GCN, 'gat': GAT, 'sage': SAGE, 'gin': GIN}.get(base_args.model_name)
                model = ModelClass(num_features=num_features, num_classes=num_tasks, hidden=args.hidden_channels,
                                   num_layers=args.num_layers, dropout=args.dropout, pool=args.pool,
                                   use_ogb_features=(base_args.dataset_type == 'ogb')).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                y_true_out, y_pred_out, _ = evaluate_graphclass(model, test_loader, device, base_args.dataset_type, evaluator)
                
                # Probability extraction
                y_probs = torch.sigmoid(y_pred_out).cpu().numpy().squeeze() if num_tasks == 1 else \
                          torch.softmax(y_pred_out, dim=1)[:, 1].cpu().numpy()
                y_true = y_true_out.cpu().numpy().squeeze()

                rank_metrics = calculate_ranking_metrics(y_true, y_probs, K_PERCENTAGES)
                for k, v in rank_metrics.items(): all_rank_metrics[k].append(v)

                # Save run metrics
                with open(os.path.join(out_dir, f'run_{run}_metrics.json'), 'w') as f:
                    json.dump(rank_metrics, f, indent=4)

            except Exception as e:
                print(f"   Run {run} failed: {e}")

        # Save summary
        summary = {f'{k}_mean': float(np.mean(v)) for k, v in all_rank_metrics.items() if v}
        summary.update({f'{k}_std': float(np.std(v)) for k, v in all_rank_metrics.items() if v})
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()




# import argparse
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import json
# import random
# from sklearn.metrics import ndcg_score
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
# from torch_geometric.transforms import OneHotDegree
# from torch_geometric.utils import degree
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# # --- Import Models & Utils ---
# from gnn_models import GCN, GAT, SAGE, GIN
# from training_utils import evaluate_graphclass  # <--- Using this to fix the error
# from torch_geometric.data.storage import GlobalStorage
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# # Fix for PyTorch > 2.6 safe globals
# torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# # =====================================================================================
# #   Configuration
# # =====================================================================================

# K_PERCENTAGES = [0.5, 1, 2, 3, 4, 5, 10]

# # =====================================================================================
# #   Ranking Metrics Helper
# # =====================================================================================

# def calculate_ranking_metrics(y_true, y_score, k_percentages):
#     y_true = np.asarray(y_true).squeeze()
#     y_score = np.asarray(y_score).squeeze()
    
#     # Sort by score descending
#     sort_indices = np.argsort(y_score)[::-1]
#     y_true_sorted = y_true[sort_indices]
    
#     total_positives = np.sum(y_true)
#     total_samples = len(y_true_sorted)
#     metrics = {}
    
#     for k_pct in k_percentages:
#         k = int(np.ceil(total_samples * (k_pct / 100.0)))
#         if k > total_samples: k = total_samples 
#         if k == 0: continue
            
#         top_k_labels = y_true_sorted[:k]
#         num_positives_in_top_k = np.sum(top_k_labels)
        
#         metric_key = f"at_{k_pct}_pct"

#         metrics[f'precision_{metric_key}'] = num_positives_in_top_k / k
#         if total_positives > 0:
#             metrics[f'recall_{metric_key}'] = num_positives_in_top_k / total_positives
#         else:
#             metrics[f'recall_{metric_key}'] = 0.0
        
#         true_relevance = y_true.reshape(1, -1)
#         pred_scores = y_score.reshape(1, -1)
#         metrics[f'ndcg_{metric_key}'] = ndcg_score(true_relevance, pred_scores, k=k)

#     return metrics

# # =====================================================================================
# #   Arg Parsing Helper
# # =====================================================================================

# def parse_param_string_to_args(param_string, base_args):
#     args = argparse.Namespace(**vars(base_args))
#     parts = param_string.split('_')
    
#     key_map = {
#         'lr': 'lr', 'pool': 'pool', 'metric': 'metric', 'hid': 'hidden_channels',
#         'do': 'dropout', 'layers': 'num_layers', 'heads': 'nhead', 'bn': 'use_bn',
#         'res': 'use_residual'
#     }
#     type_map = {
#         'lr': float, 'pool': str, 'metric': str, 'hid': int, 'do': float,
#         'layers': int, 'heads': int, 'bn': lambda x: x.lower() == 'true',
#         'res': lambda x: x.lower() == 'true'
#     }
#     if 'class_weight' in parts: args.use_class_weight = True
#     for part in parts:
#         if '-' not in part: continue
#         try:
#             key_short, val = part.split('-', 1)
#             if key_short in key_map:
#                 setattr(args, key_map[key_short], type_map[key_short](val))
#         except: pass
#     return args

# def fix_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # =====================================================================================
# #   Main Execution
# # =====================================================================================

# def main():
#     parser = argparse.ArgumentParser(description='Fast Top-K Evaluation')
    
#     parser.add_argument('--dataset', type=str, required=True)
#     parser.add_argument('--model_name', type=str, required=True)
#     parser.add_argument('--runs', type=int, default=3)
#     parser.add_argument('--filter', type=str, default=None)
#     parser.add_argument('--base_results_dir', type=str, default='results_weighted')
    
#     # Defaults
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--hidden_channels', type=int, default=300)
#     parser.add_argument('--num_layers', type=int, default=5)
#     parser.add_argument('--dropout', type=float, default=0.5)
#     parser.add_argument('--pool', type=str, default='mean')
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--nhead', type=int, default=1)
#     parser.add_argument('--use_bn', action='store_true')
#     parser.add_argument('--use_residual', action='store_true')
#     parser.add_argument('--seed', type=int, default=42)

#     base_args = parser.parse_args()
#     base_args.dataset_type = 'ogb' if 'ogb' in base_args.dataset else 'tu'
#     device = torch.device(f"cuda:{base_args.device}" if torch.cuda.is_available() else "cpu")
#     fix_seed(base_args.seed)

#     # 1. Load Data
#     print(f"--- Loading {base_args.dataset} ---")
#     if base_args.dataset_type == 'ogb':
#         dataset = PygGraphPropPredDataset(name=base_args.dataset, root='data/OGB')
#         split_idx = dataset.get_idx_split()
#         test_loader = DataLoader(dataset[split_idx["test"]], batch_size=base_args.batch_size, shuffle=False)
#         evaluator = Evaluator(name=base_args.dataset)
#         num_tasks = dataset.num_tasks
#         num_features = dataset.num_features
#     else:
#         dataset = TUDataset(root='data/TUDataset', name=base_args.dataset)
#         if dataset.num_node_features == 0:
#             max_deg = max(int(degree(g.edge_index[0], num_nodes=g.num_nodes).max()) for g in dataset)
#             dataset.transform = OneHotDegree(max_degree=max_deg)
#         indices = list(range(len(dataset)))
#         random.shuffle(indices) 
#         test_size = len(dataset) // 10
#         test_dataset = dataset[indices[:test_size]]
#         test_loader = DataLoader(test_dataset, batch_size=base_args.batch_size, shuffle=False)
#         evaluator = None
#         num_tasks = dataset.num_classes
#         num_features = dataset.num_features

#     # 2. Locate Experiments
#     base_path = os.path.join(base_args.base_results_dir, base_args.dataset, base_args.model_name)
#     if not os.path.exists(base_path):
#         print(f"Directory not found: {base_path}")
#         return

#     dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
#     dirs = [d for d in dirs if d not in ['summary', 'topk_percent', 'topk_results'] and not d.startswith('.')]
#     if base_args.filter:
#         dirs = [d for d in dirs if base_args.filter in d]
    
#     print(f"Found {len(dirs)} experiments to process.")

#     # 3. Process Experiments
#     for folder in dirs:
#         print(f"\nProcessing: {folder}")
#         exp_path = os.path.join(base_path, folder)
        
#         # Output folder: topk_results
#         out_dir = os.path.join(exp_path, 'topk_results')
#         os.makedirs(out_dir, exist_ok=True)

#         try:
#             args = parse_param_string_to_args(folder, base_args)
#         except:
#             continue

#         all_rank_metrics = {f'{m}_at_{k}_pct': [] for k in K_PERCENTAGES for m in ['precision', 'recall', 'ndcg']}

#         for run in range(1, args.runs + 1):
#             run_dir = os.path.join(exp_path, f'run_{run}')
#             model_path = os.path.join(run_dir, 'model.pt')
            
#             if not os.path.exists(model_path): continue

#             ModelClass = {'gcn': GCN, 'gat': GAT, 'sage': SAGE, 'gin': GIN}.get(base_args.model_name)
#             if not ModelClass: continue

#             try:
#                 # GNN Args
#                 gnn_args = {
#                     'num_features': num_features, 'num_classes': num_tasks, 'hidden': args.hidden_channels,
#                     'num_layers': args.num_layers, 'dropout': args.dropout, 'pool': args.pool,
#                     'use_ogb_features': (base_args.dataset_type == 'ogb'), 
#                     'use_bn': args.use_bn, 'use_residual': args.use_residual
#                 }
#                 # if base_args.model_name == 'gat': gnn_args['nhead'] = args.nhead

#                 model = ModelClass(**gnn_args)
#                 model.load_state_dict(torch.load(model_path, map_location=device))
#                 model.to(device)
#                 model.eval()

#                 # --- FIX: Use evaluate_graphclass for correct model inference ---
#                 y_test_true, y_test_pred, _ = evaluate_graphclass(model, test_loader, device, base_args.dataset_type, evaluator)
                
#                 # Convert logits to numpy/probs
#                 if num_tasks == 1:
#                     y_probs = torch.sigmoid(y_test_pred).cpu().numpy().squeeze()
#                 else:
#                     probs = torch.softmax(y_test_pred, dim=1)
#                     y_probs = probs[:, 1].cpu().numpy()
                
#                 y_true = y_test_true.cpu().numpy().squeeze()

#                 # Calculate New Metrics
#                 rank_metrics = calculate_ranking_metrics(y_true, y_probs, K_PERCENTAGES)
#                 for k, v in rank_metrics.items(): all_rank_metrics[k].append(v)

#                 # --- NON-INTRUSIVE SAVE ---
#                 # 1. Read original metrics to keep validation scores
#                 orig_metrics_path = os.path.join(run_dir, 'metrics.json')
#                 run_data = {}
#                 if os.path.exists(orig_metrics_path):
#                     with open(orig_metrics_path, 'r') as f:
#                         run_data = json.load(f)
                
#                 # 2. Update with new top-k data
#                 run_data.update(rank_metrics)
                
#                 # 3. Save to NEW file in topk_results (run_X_metrics.json)
#                 # Note: analysis script expects 'metrics.json' inside 'run_X' usually.
#                 # To make this work with analyze_results.py without modifying run folders,
#                 # we must either modify analyze_results.py to look in topk_results,
#                 # OR we accept modifying the run folder's metrics.json.
#                 # Your prompt said: "do what you have to do in topk_results folder do not touch other folders".
#                 # I will save as 'run_{run}_metrics.json' in 'topk_results'.
#                 # NOTE: You will need to point analyze_results.py to look here later or manually copy.
                
#                 new_metrics_path = os.path.join(out_dir, f'run_{run}_metrics.json')
#                 with open(new_metrics_path, 'w') as f:
#                     json.dump(run_data, f, indent=4)

#             except Exception as e:
#                 print(f"  Run {run} failed: {e}")
#                 continue

#         # Save Summary to topk_results
#         summary = {}
#         for key, values in all_rank_metrics.items():
#             if values:
#                 summary[f'{key}_mean'] = np.mean(values)
#                 summary[f'{key}_std'] = np.std(values)
        
#         with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
#             json.dump(summary, f, indent=4)
            
#     print("\nDone. Results saved in 'topk_results' folder.")

# if __name__ == "__main__":
#     main()