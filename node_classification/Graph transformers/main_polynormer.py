# Save this file as main_polynormer.py
# This is a dedicated training script for the Polynormer model.

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
import json
from sklearn.preprocessing import label_binarize

# Import necessary utilities from your project structure
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, class_rand_splits, load_fixed_splits
from eval import evaluate
from logger import Logger
from parse import parser_add_main_args

# Import our adapted Polynormer model
from polynormer_adapted import PolynormerAdapted

def fix_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_results(model, embeddings, tsne_embeds, tsne_labels, logits, labels, args, out_dir):
    """Saves all artifacts from a training run."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings.cpu().numpy())
    np.save(os.path.join(out_dir, 'tsne_embeds.npy'), tsne_embeds)
    np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
    np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
    np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

def plot_tsne(tsne_embeds, tsne_labels, out_dir):
    """Generates and saves a t-SNE plot of node embeddings."""
    plt.figure(figsize=(8, 6))
    for c in np.unique(tsne_labels):
        idx = tsne_labels == c
        plt.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=f'Class {c}', alpha=0.6)
    plt.legend()
    plt.title('t-SNE of Node Embeddings (Polynormer)')
    plt.savefig(os.path.join(out_dir, 'tsne_plot.png'))
    plt.close()

def plot_logits_distribution(logits, labels, out_dir):
    """Generates and saves a histogram of the model's output logits."""
    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().numpy().squeeze()
    num_classes = logits.shape[1]
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        if np.any(labels == c):
            plt.hist(logits[labels == c, c], bins=30, alpha=0.5, label=f'Class {c}')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.title('Logits Distribution per Class (Polynormer)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'logits_distribution.png'))
    plt.close()
    
def train_and_evaluate(args):
    """Main function to train and evaluate the Polynormer model."""
    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    args.device = device
    print(f"Using device: {device}")

    # Set model name for directory saving. This script is only for polynormer.
    args.gnn = 'polynormer'

    # --- Data Loading and Preprocessing (from GNN script) ---
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f'Dataset: {args.dataset}, Nodes: {n}, Features: {d}, Classes: {c}')

    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

    # --- Model Initialization (Specific to Polynormer) ---
    print("Instantiating PolynormerAdapted model...")
    model = PolynormerAdapted(
        in_channels=d,
        hidden_channels=args.hidden_channels,
        out_channels=c,
        local_layers=args.local_layers,
        global_layers=args.global_layers, # Polynormer-specific argument
        dropout=args.dropout,
        heads=args.num_heads,
        pre_ln=args.pre_ln
    ).to(device)
    
    # --- Loss, Optimizer, and Evaluation Setup (from GNN script) ---
    criterion = nn.NLLLoss() if args.dataset not in ('questions') else nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc if args.metric == 'rocauc' else eval_acc
    logger = Logger(args.runs, args)
    
    all_acc, all_roc_auc, all_pr_auc, all_reports = [], [], [], []

    exp_name = f"l_{args.local_layers}-h_{args.hidden_channels}"

    # --- Training & Evaluation Loop (from GNN script) ---
    for run in range(args.runs):
        split_idx = split_idx_lst[0] if args.dataset in ('cora', 'citeseer', 'pubmed') else split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        best_val, best_test = float('-inf'), float('-inf')
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            # Conditional loss logic for different dataset types
            if args.dataset in ('questions'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, num_classes=c).squeeze(1)
                else:
                    true_label = dataset.label
                loss = criterion(out[train_idx], true_label[train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            
            loss.backward()
            optimizer.step()

            result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val, best_test = result[1], result[2]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % args.display_step == 0:
                print(f'Run: {run+1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100*result[0]:.2f}%, Valid: {100*result[1]:.2f}%, Test: {100*result[2]:.2f}%')
        
        print(f'Run {run+1:02d} finished. Best Valid: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')
        model.load_state_dict(best_model_state)
        
        # --- Final Evaluation, Visualization, and Saving (from GNN script) ---
        model.eval()
        with torch.no_grad():
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

        test_idx = split_idx['test'].to(device)
        y_true_test = dataset.label[test_idx]

        if args.dataset not in ('questions'):
            out = F.log_softmax(out, dim=1)
        
        probs = torch.exp(out[test_idx]) if args.dataset not in ('questions') else torch.sigmoid(out[test_idx])
        y_pred = out.argmax(dim=-1, keepdim=True)
        
        accuracy = eval_acc(y_true_test, out[test_idx])
        report = classification_report(y_true_test.cpu().numpy(), y_pred[test_idx].cpu().numpy(), output_dict=True, zero_division=0)
        
        if c > 2:
            roc_auc = roc_auc_score(y_true_test.cpu().numpy(), probs.cpu().numpy(), multi_class='ovr')
            y_true_binarized = label_binarize(y_true_test.cpu().numpy(), classes=range(c))
            pr_auc = average_precision_score(y_true_binarized, probs.cpu().numpy())
        else:
            roc_auc = eval_rocauc(y_true_test, out[test_idx])
            pr_auc = average_precision_score(y_true_test.cpu().numpy(), probs[:,1].cpu().numpy())

        print(f'Final Test Accuracy: {accuracy:.4f}, AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}')
        all_acc.append(accuracy); all_roc_auc.append(roc_auc); all_pr_auc.append(pr_auc); all_reports.append(report)

        with torch.no_grad():
            embeddings = model.get_embeddings(dataset.graph['node_feat'], dataset.graph['edge_index'])
            embeddings_cpu = embeddings[test_idx].cpu().numpy()
            labels_cpu = y_true_test.cpu().numpy().squeeze()
            tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(labels_cpu)-1))
            tsne_embeds = tsne.fit_transform(embeddings_cpu)

        out_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/run_{run+1}'
        save_results(model, embeddings[test_idx], tsne_embeds, labels_cpu, out[test_idx], y_true_test, args, out_dir)
        plot_tsne(tsne_embeds, labels_cpu, out_dir)
        plot_logits_distribution(out[test_idx], y_true_test, out_dir)

        metrics = {'accuracy': accuracy, 'auc_roc': roc_auc, 'auc_pr': pr_auc, 'classification_report': report}
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    # --- Summary Over All Runs (from GNN script) ---
    summary = {
        'accuracy_mean': float(np.mean(all_acc)), 'accuracy_std': float(np.std(all_acc)),
        'auc_roc_mean': float(np.mean(all_roc_auc)), 'auc_roc_std': float(np.std(all_roc_auc)),
        'auc_pr_mean': float(np.mean(all_pr_auc)), 'auc_pr_std': float(np.std(all_pr_auc)),
    }
    summary_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/'
    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary over {args.runs} runs:\n", json.dumps(summary, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dedicated Training Pipeline for Polynormer')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    train_and_evaluate(args)