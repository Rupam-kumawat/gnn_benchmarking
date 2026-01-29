# =====================================================================================
#  Integrated GNN Training Pipeline

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, to_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, balanced_accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
import json
import scipy.sparse as sp
import networkx as nx

# --- Model Imports ---
from model import MPNNs
# from FSGCN_models import FSGNN
# from glognn_models import MLP_NORM
# from gprgnn_models import GPRGNN
from mlp_model import MLP
# from xgboost_model import xgboost

# --- Helper Imports ---
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, eval_prauc, class_rand_splits, load_fixed_splits, eval_balanced_acc
from eval import evaluate
from logger import Logger
from parse import parser_add_main_args


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

# def save_results(model, embeddings, tsne_embeds, tsne_labels, logits, labels, args, out_dir):
#     """Saves all run artifacts to a directory."""
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
#     np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings.cpu().numpy())
#     np.save(os.path.join(out_dir, 'tsne_embeds.npy'), tsne_embeds)
#     np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
#     np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
#     np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

def save_results(model, tsne_labels, logits, labels, args, out_dir):
    """Saves all run artifacts to a directory."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    # np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings.cpu().numpy())
    # np.save(os.path.join(out_dir, 'tsne_embeds.npy'), tsne_embeds)
    np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
    np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
    np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())
    
    # save_results(model, labels_cpu, out[split_idx['test']], y_true[split_idx['test']], args, out_dir)

def plot_tsne(tsne_embeds, tsne_labels, out_dir):
    """Generates and saves a t-SNE plot."""
    plt.figure(figsize=(8, 6))
    for c in np.unique(tsne_labels):
        idx = tsne_labels == c
        plt.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=f'Class {c}', alpha=0.6)
    plt.legend()
    plt.title('t-SNE of Node Embeddings')
    plt.savefig(os.path.join(out_dir, 'tsne_plot.png'))
    plt.close()

def plot_logits_distribution(logits, labels, out_dir):
    """Generates and saves a plot of the logits distribution."""
    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().numpy().squeeze()
    num_classes = logits.shape[1]
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        plt.hist(logits[labels == c, c], bins=30, alpha=0.5, label=f'Class {c}')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.title('Logits Distribution per Class')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'logits_distribution.png'))
    plt.close()
    
def generate_param_string(args):
    """
    Generates a unique, readable string from all relevant hyperparameters.
    """
    params = []
    
    # --- General Training Hyperparameters ---
    params.append(f"metric-{args.metric}")
    params.append(f"lr-{args.lr}")
    params.append(f"wd-{args.weight_decay}")
    params.append(f"do-{args.dropout}")
    params.append(f"hid-{args.hidden_channels}")

    # --- Model-Specific Hyperparameters ---
    gnn_type = args.gnn
    
    if gnn_type in ['gcn', 'gat', 'sage', 'gin']:
        params.append(f"layers-{args.local_layers}")
        if gnn_type == 'gat':
            params.append(f"heads-{args.num_heads}")
        # Add boolean flags if they are true
        if args.res: params.append('res')
        if args.ln: params.append('ln')
        if args.bn: params.append('bn')
        if args.jk: params.append('jk')

    elif gnn_type == 'fsgcn':
        params.append(f"layers-{args.fsgcn_num_layers}")
        params.append(f"type-{args.fsgcn_feat_type}")
        if args.fsgcn_layer_norm: params.append('ln')

    elif gnn_type == 'glognn':
        # Correctly includes all GloGNN parameters
        params.append(f"alpha-{args.glognn_alpha}")
        params.append(f"beta-{args.glognn_beta}")
        params.append(f"gamma-{args.glognn_gamma}")
        params.append(f"delta-{args.glognn_delta}")
        params.append(f"normfunc-{args.glognn_norm_func_id}")
        params.append(f"normlayers-{args.glognn_norm_layers}")
        params.append(f"orders-{args.glognn_orders}")
        params.append(f"orderfunc-{args.glognn_orders_func_id}")

    elif gnn_type == 'gprgnn':
        # Correctly includes all GPRGNN parameters
        params.append(f"ppnp-{args.gprgnn_ppnp}")
        params.append(f"K-{args.gprgnn_K}")
        params.append(f"alpha-{args.gprgnn_alpha}")
        params.append(f"dprate-{args.gprgnn_dprate}")
        params.append(f"init-{args.gprgnn_Init}")
        
    return "_".join(params)

# =====================================================================================
#  Section 2: Main Training and Evaluation Function
# =====================================================================================
        
def train_and_evaluate(args):
    """Main function to run the training and evaluation pipeline."""
    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('device', device)

    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)
    
    n, c, d = dataset.graph['num_nodes'], max(dataset.label.max().item() + 1, dataset.label.shape[1]), dataset.graph['node_feat'].shape[1]
    print(f'Dataset: {args.dataset}, Nodes: {n}, Classes: {c}, Features: {d}')
    
    # --- Pre-computation and Model Initialization ---
    fsgcn_list_mat = None
    glognn_adj_sparse, glognn_adj_dense, glognn_features = None, None, None
    gprgnn_features = None

    if args.gnn == 'fsgcn':
        print("--- FSGCN: Starting feature pre-computation ---")
        edge_index_no_loops, _ = remove_self_loops(dataset.graph['edge_index'])
        adj = to_scipy_sparse_matrix(edge_index_no_loops, num_nodes=n)
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

        edge_index_with_loops, _ = add_self_loops(edge_index_no_loops, num_nodes=n)
        adj_i = to_scipy_sparse_matrix(edge_index_with_loops, num_nodes=n)
        adj_i = sparse_mx_to_torch_sparse_tensor(adj_i).to(device)

        list_mat = [dataset.graph['node_feat'].to(device)]
        no_loop_mat = loop_mat = dataset.graph['node_feat'].to(device)

        for _ in range(args.fsgcn_num_layers):
            no_loop_mat = torch.spmm(adj, no_loop_mat)
            loop_mat = torch.spmm(adj_i, loop_mat)
            list_mat.append(no_loop_mat)
            list_mat.append(loop_mat)

        if args.fsgcn_feat_type == "homophily":
            select_idx = [0] + [2 * ll for ll in range(1, args.fsgcn_num_layers + 1)]
            fsgcn_list_mat = [list_mat[ll] for ll in select_idx]
        elif args.fsgcn_feat_type == "heterophily":
            select_idx = [0] + [2 * ll - 1 for ll in range(1, args.fsgcn_num_layers + 1)]
            fsgcn_list_mat = [list_mat[ll] for ll in select_idx]
        else:
            fsgcn_list_mat = list_mat
        
        model = FSGNN(nlayers=len(fsgcn_list_mat), nfeat=d, nhidden=args.hidden_channels, nclass=c,
                      dropout=args.dropout, layer_norm=args.fsgcn_layer_norm).to(device)
        print(f"--- FSGCN: Pre-computation finished. Using {len(fsgcn_list_mat)} feature matrices. ---")
        
    elif args.gnn == 'glognn':
        print("--- GloGNN: Pre-computation started ---")
        glognn_features = dataset.graph['node_feat'].to(device)
        
        edge_index_np = dataset.graph['edge_index'].cpu().numpy().T
        adj = nx.adjacency_matrix(nx.from_edgelist(edge_index_np), nodelist=range(n))
        glognn_adj_sparse = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        glognn_adj_dense = glognn_adj_sparse.to_dense().to(device)

        model = MLP_NORM(nnodes=n, nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout,
                         alpha=args.glognn_alpha, beta=args.glognn_beta, gamma=args.glognn_gamma,
                         delta=args.glognn_delta, norm_func_id=args.glognn_norm_func_id,
                         norm_layers=args.glognn_norm_layers, orders=args.glognn_orders,
                         orders_func_id=args.glognn_orders_func_id, device=device).to(device)
        print("--- GloGNN: Pre-computation finished ---")

    elif args.gnn == 'gprgnn':
        print("--- GPRGNN: Pre-computation started ---")
        features_np = dataset.graph['node_feat'].numpy()
        normalized_features = normalize_tensor_sparse(sp.csr_matrix(features_np), symmetric=0).todense()
        gprgnn_features = torch.FloatTensor(normalized_features).to(device)

        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
        
        model = GPRGNN(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout,
                       dprate=args.gprgnn_dprate, K=args.gprgnn_K, alpha=args.gprgnn_alpha,
                       Init=args.gprgnn_Init, ppnp=args.gprgnn_ppnp).to(device)
        print("--- GPRGNN: Pre-computation finished ---")
    elif args.gnn == 'mlp':
        print("--- MLP: Initializing Model ---")
        dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
        model = MLP(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout).to(device)
    
    # elif args.gnn == 'xgboost':
    #     print("--- Xgboost: Initializing Model ---")
    #     dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    #     model = xgboost(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout).to(device)
        
        
    else: # Standard PyG GNNs
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
        dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
        dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
        
        model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
                      heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
                      res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn=args.gnn).to(device)

    criterion = nn.BCEWithLogitsLoss() if args.dataset in ('questions') else nn.NLLLoss()
    eval_func = {'prauc': eval_prauc, 'rocauc': eval_rocauc, 'balacc': eval_balanced_acc}.get(args.metric, eval_acc)
    logger = Logger(args.runs, args)
    
    all_acc, vals, all_balanced_acc, all_roc_auc, all_pr_auc = [], [], [], [], []
    all_reports_default, all_reports_optimal = [], []

    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        
        # --- Conditional Optimizer Setup ---
        if args.gnn == 'gprgnn':
            optimizer = torch.optim.Adam([
                {'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.lr}
            ], lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_val, best_test = float('-inf'), float('-inf')
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            
            # --- Conditional Forward Pass ---
            if args.gnn == 'fsgcn':
                out = model(fsgcn_list_mat)
            elif args.gnn == 'glognn':
                out = model(glognn_features, glognn_adj_sparse, glognn_adj_dense)
            elif args.gnn == 'gprgnn':
                out = model(gprgnn_features, dataset.graph['edge_index'])
            elif args.gnn == 'mlp':
                out = model(dataset.graph['node_feat'])
            # elif args.gnn == 'xgboost':
            #     out = model(dataset.graph['node_feat'])
            else: # Standard GNNs
                out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            if args.dataset in ('questions'):
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1) if dataset.label.shape[1] == 1 else dataset.label
                loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
            else:
                out_log_softmax = F.log_softmax(out, dim=1)
                loss = criterion(out_log_softmax[train_idx], dataset.label.squeeze(1)[train_idx])

            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args, 
                                  fsgcn_list_mat=fsgcn_list_mat, 
                                  glognn_features=glognn_features, glognn_adj_sparse=glognn_adj_sparse, glognn_adj_dense=glognn_adj_dense,
                                  gprgnn_features=gprgnn_features)
            
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val, best_test = result[1], result[2]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100*result[0]:.2f}%, Valid: {100*result[1]:.2f}%, Test: {100*result[2]:.2f}%')

        print(f'Run {run+1}/{args.runs}: Best Valid: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')

        # --- Final Evaluation ---
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            if args.gnn == 'fsgcn':
                out = model(fsgcn_list_mat)
            elif args.gnn == 'glognn':
                out = model(glognn_features, glognn_adj_sparse, glognn_adj_dense)
            elif args.gnn == 'gprgnn':
                out = model(gprgnn_features, dataset.graph['edge_index'])
            elif args.gnn == 'mlp':
                out = model(dataset.graph['node_feat'])
            # elif args.gnn == 'xgboost':
            #     out = model(dataset.graph['node_feat'])
            else:
                out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

        y_true, test_idx = dataset.label, split_idx['test']
        best_threshold, report_optimal = 0.5, None

        if c > 2: # Multi-class
            probs = torch.exp(F.log_softmax(out[test_idx], dim=1))
            y_pred = probs.argmax(dim=-1)
            roc_auc = roc_auc_score(y_true[test_idx].cpu().numpy(), probs.detach().cpu().numpy(), multi_class='ovr')
            pr_auc = eval_prauc(y_true[test_idx], F.log_softmax(out[test_idx], dim=1))
            balanced_acc = balanced_accuracy_score(y_true[test_idx].cpu().numpy(), y_pred.cpu().numpy())
            report_default = classification_report(y_true[test_idx].cpu().numpy(), y_pred.cpu().numpy(), output_dict=True)
        else: # Binary
            out_log_softmax = F.log_softmax(out, dim=1)
            roc_auc = eval_rocauc(y_true[test_idx], out_log_softmax[test_idx])
            probs_test = torch.exp(out_log_softmax[test_idx])
            pr_auc = average_precision_score(y_true[test_idx].cpu().numpy(), probs_test[:,1].detach().cpu().numpy())

            valid_idx = split_idx['valid']
            valid_probs = torch.exp(out_log_softmax[valid_idx])[:, 1].cpu().numpy()
            valid_true = y_true[valid_idx].cpu().numpy()
            best_f1 = -1
            for threshold in np.linspace(0, 1, 100):
                f1 = f1_score(valid_true, (valid_probs >= threshold).astype(int), average='macro')
                if f1 > best_f1:
                    best_f1, best_threshold = f1, threshold
            print(f'Optimal Threshold on Val Set: {best_threshold:.4f} (Macro F1: {best_f1:.4f})')
            
            y_true_test_np = y_true[test_idx].cpu().numpy()
            y_pred_default = out_log_softmax[test_idx].argmax(dim=-1).cpu().numpy()
            y_pred_optimal = (probs_test[:, 1].cpu().numpy() >= best_threshold).astype(int)
            balanced_acc = balanced_accuracy_score(y_true_test_np, y_pred_default)
            report_default = classification_report(y_true_test_np, y_pred_default, output_dict=True)
            report_optimal = classification_report(y_true_test_np, y_pred_optimal, output_dict=True)

        print(f'AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}, Balanced Acc: {balanced_acc:.4f}')
        print('\n--- Classification Report (Default Threshold) ---\n', json.dumps(report_default, indent=2))
        if report_optimal: print('\n--- Classification Report (Optimal Threshold) ---\n', json.dumps(report_optimal, indent=2))

        all_acc.append(best_test)
        vals.append(best_val)
        all_balanced_acc.append(balanced_acc)
        all_roc_auc.append(roc_auc)
        all_pr_auc.append(pr_auc)
        all_reports_default.append(report_default)
        if report_optimal: all_reports_optimal.append(report_optimal)

        # # --- Embedding Extraction and Saving ---
        with torch.no_grad():
        #     if args.gnn == 'fsgcn':
        #         embeddings = model.get_embeddings(fsgcn_list_mat)
        #     elif args.gnn == 'glognn':
        #         embeddings = model.get_embeddings(glognn_features, glognn_adj_sparse, glognn_adj_dense)
        #     elif args.gnn == 'gprgnn':
        #         embeddings = model.get_embeddings(gprgnn_features)
        #     elif args.gnn == 'mlp':
        #         embeddings = model.get_embeddings(dataset.graph['node_feat'])
        #     # elif args.gnn == 'xgboost':
        #     #     embeddings = model.get_embeddings(dataset.graph['node_feat'])
        #     else:
        #         embeddings = model.get_embeddings(dataset.graph['node_feat'], dataset.graph['edge_index'])
                
        #     embeddings_cpu = embeddings[split_idx['test']].cpu()
            labels_cpu = y_true[split_idx['test']].cpu().numpy().squeeze()
        #     tsne = TSNE(n_components=2, random_state=42)
        #     tsne_embeds = tsne.fit_transform(embeddings_cpu.numpy())

        # out_dir = f'{args.dataset}_results_{run}_{args.gnn}_{args.metric}'
        param_string = generate_param_string(args)
        out_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/run_{run}'
        
        # save_results(model, embeddings[split_idx['test']], tsne_embeds, labels_cpu, out[split_idx['test']], y_true[split_idx['test']], args, out_dir)
        # save_results(model, labels_cpu, out[split_idx['test']], y_true[split_idx['test']], args, out_dir)
        print('save results started')
        save_results(model, labels_cpu, out[split_idx['test']], y_true[split_idx['test']], args, out_dir)
        # plot_tsne(tsne_embeds, labels_cpu, out_dir)
        # plot_logits_distribution(out[split_idx['test']], y_true[split_idx['test']], out_dir)
        print('save results finished')

        metrics = {'accuracy_vals': best_val,
            'accuracy_from_val': best_test, 'balanced_accuracy': balanced_acc, 'auc_roc': roc_auc, 'auc_pr': pr_auc,
            'optimal_threshold': best_threshold if c <= 2 else None,
            'classification_report_default': report_default, 'classification_report_optimal': report_optimal,
        }
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
            
    # --- Final Summary ---
    summary = {
        'accuracy_mean_vals': float(np.mean(vals)), 'accuracy_std_vals': float(np.std(vals)),
        'accuracy_mean': float(np.mean(all_acc)), 'accuracy_std': float(np.std(all_acc)),
        'balanced_accuracy_mean': float(np.mean(all_balanced_acc)), 'balanced_accuracy_std': float(np.std(all_balanced_acc)),
        'auc_roc_mean': float(np.mean(all_roc_auc)), 'auc_roc_std': float(np.std(all_roc_auc)),
        'auc_pr_mean': float(np.mean(all_pr_auc)), 'auc_pr_std': float(np.std(all_pr_auc)),
    }
    # summary_dir = f'{args.dataset}_results_summary_{args.gnn}_{args.metric}'
    
    param_string = generate_param_string(args) # Or re-use from the last run
    summary_dir = f'results/{args.dataset}/{args.gnn}/{param_string}/summary'
    
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(summary_dir, 'reports_default.json'), 'w') as f:
        json.dump(all_reports_default, f, indent=2)
    if all_reports_optimal:
        with open(os.path.join(summary_dir, 'reports_optimal.json'), 'w') as f:
            json.dump(all_reports_optimal, f, indent=2)

# =====================================================================================
#  Section 3: Script Entry Point
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrated GNN Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    train_and_evaluate(args)
