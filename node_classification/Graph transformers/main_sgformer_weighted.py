# Save this file as main_sgformer.py
# Dedicated training script for SGFormer: Focal Loss + Validation Logging + Flexible Dir Logic.

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

# Import our adapted SGFormer model
from sgformer_adapted import SGFormerAdapted

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def fix_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_artifacts(model, embeddings, tsne_embeds, tsne_labels, out_dir):
    """Saves model weights and visualization data."""
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings.cpu().detach().numpy())
    np.save(os.path.join(out_dir, 'tsne_embeds.npy'), tsne_embeds)
    np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)

def plot_visualizations(tsne_embeds, tsne_labels, logits, labels, out_dir):
    # t-SNE Plot
    plt.figure(figsize=(8, 6))
    for c in np.unique(tsne_labels):
        idx = tsne_labels == c
        plt.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=f'Class {c}', alpha=0.6)
    plt.legend(); plt.title('t-SNE of Node Embeddings (SGFormer)'); plt.savefig(os.path.join(out_dir, 'tsne_plot.png')); plt.close()

    # Logits Distribution
    logits_np = logits.cpu().detach().numpy()
    labels_np = labels.cpu().numpy().squeeze()
    plt.figure(figsize=(8, 6))
    for c in range(logits_np.shape[1]):
        if np.any(labels_np == c):
            plt.hist(logits_np[labels_np == c, c], bins=30, alpha=0.5, label=f'Class {c}')
    plt.xlabel('Logit Value'); plt.ylabel('Frequency'); plt.legend(); plt.savefig(os.path.join(out_dir, 'logits_dist.png')); plt.close()

def train_and_evaluate(args):
    """Main function to train and evaluate the SGFormer model."""
    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    args.device = device
    print(f"Using device: {device}")

    # Set model name for directory saving
    args.gnn = 'sgformer'

    # --- Data Loading and Preprocessing ---
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

    # --- Model Initialization ---
    print("Instantiating SGFormerAdapted model...")
    model = SGFormerAdapted(
        in_channels=d,
        hidden_channels=args.hidden_channels,
        out_channels=c,
        local_layers=args.local_layers,
        dropout=args.dropout,
        heads=args.num_heads,
        res=args.res,
        ln=args.ln,
        jk=args.jk
    ).to(device)
    
    # --- Loss Setup ---
    if args.use_focal:
        print(f"Using Focal Loss with gamma={args.focal_gamma}")
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        print("Using Standard Loss (NLL or BCE)")
        criterion = nn.NLLLoss() if args.dataset not in ('questions') else nn.BCEWithLogitsLoss()

    eval_func = eval_rocauc if args.metric == 'rocauc' else eval_acc
    logger = Logger(args.runs, args)
    
    all_runs = []
    exp_name = f"l_{args.layers}-h_{args.hidden_channels}-focal_{args.use_focal}-gw_{args.graph_weight}"

    # --- Training & Evaluation Loop ---
    for run in range(args.runs):
        split_idx = split_idx_lst[0] if args.dataset in ('cora', 'citeseer', 'pubmed') else split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        best_val, best_test = float('-inf'), float('-inf')
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            if args.dataset in ('questions'):
                true_label = F.one_hot(dataset.label, num_classes=c).squeeze(1) if dataset.label.shape[1] == 1 else dataset.label
                loss = criterion(out[train_idx], true_label[train_idx].to(torch.float))
            else:
                if args.use_focal:
                    loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
                else:
                    out_log = F.log_softmax(out, dim=1)
                    loss = criterion(out_log[train_idx], dataset.label.squeeze(1)[train_idx])
            
            loss.backward()
            optimizer.step()

            result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            val_loss = result[3]
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val, best_test = result[1], result[2]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % args.display_step == 0:
                print(f'Run: {run+1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Valid: {100*result[1]:.2f}%, Test: {100*result[2]:.2f}%')
            scheduler.step(val_loss)
        
        # Load best and finalize run
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
            out_log = F.log_softmax(out, dim=1) if args.dataset not in ('questions') else out
            test_idx = split_idx['test'].to(device)
            y_true_test = dataset.label[test_idx]
            y_pred = out.argmax(dim=-1, keepdim=True)
            probs = torch.exp(out_log[test_idx]) if args.dataset not in ('questions') else torch.sigmoid(out[test_idx])

            # Metrics Calculation
            acc = eval_acc(y_true_test, out_log[test_idx])
            if c > 2:
                roc_auc = roc_auc_score(y_true_test.cpu().numpy(), probs.cpu().numpy(), multi_class='ovr')
                pr_auc = average_precision_score(label_binarize(y_true_test.cpu().numpy(), classes=range(c)), probs.cpu().numpy())
            else:
                roc_auc = eval_rocauc(y_true_test, out_log[test_idx])
                pr_auc = average_precision_score(y_true_test.cpu().numpy(), probs[:,1].cpu().numpy() if probs.dim() > 1 else probs.cpu().numpy())
            
            report = classification_report(y_true_test.cpu().numpy(), y_pred[test_idx].cpu().numpy(), output_dict=True, zero_division=0)

        # 🔹 Output Directory Logic
        if args.output_dir is not None:
            out_dir = os.path.join(args.output_dir, f'run_{run+1}')
        else:
            out_dir = f'results/{args.dataset}/{args.gnn}/{exp_name}/run_{run+1}'
        os.makedirs(out_dir, exist_ok=True)

        run_results = {
            "val_metric": best_val, "test_acc": acc, "test_roc_auc": roc_auc, 
            "test_pr_auc": pr_auc, "report": report
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(run_results, f, indent=2)

        # Artifacts and Visualization
        embeddings = model.get_embeddings(dataset.graph['node_feat'], dataset.graph['edge_index'])
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(test_idx)-1))
        tsne_embeds = tsne.fit_transform(embeddings[test_idx].cpu().detach().numpy())
        save_artifacts(model, embeddings[test_idx], tsne_embeds, y_true_test.cpu().numpy().squeeze(), out_dir)
        plot_visualizations(tsne_embeds, y_true_test.cpu().numpy().squeeze(), out[test_idx], y_true_test, out_dir)

        all_runs.append(run_results)
        print(f'Run {run+1:02d} finished. Val: {100*best_val:.2f}%, Test: {100*acc:.2f}%')

    # ===== SUMMARY =====
    summary = {}
    numeric_keys = [k for k, v in all_runs[0].items() if isinstance(v, (int, float))]
    for k in numeric_keys:
        values = [r[k] for r in all_runs]
        summary[f"{k}_mean"] = float(np.mean(values))
        summary[f"{k}_std"] = float(np.std(values))

    summary_dir = args.output_dir if args.output_dir is not None else f'results/{args.dataset}/{args.gnn}/{exp_name}'
    os.makedirs(summary_dir, exist_ok=True)

    with open(os.path.join(summary_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(summary_dir, "all_runs.json"), "w") as f:
        json.dump(all_runs, f, indent=2)

    print(f"\nResults saved to: {summary_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGFormer Training with Focal Loss and Val Logging')
    parser_add_main_args(parser)
    parser.add_argument('--use_focal', action='store_true', help='Use Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output path')
    args = parser.parse_args()
    print(args)
    train_and_evaluate(args)