


import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, balanced_accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
import json

# --- Model Imports ---
from model import MPNNs
from mlp_model import MLP

# --- Helper Imports ---
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, eval_prauc, class_rand_splits, load_fixed_splits, eval_balanced_acc
from logger import Logger
from parse import parser_add_main_args


# =====================================================================================
#  Section 1: Custom Loss Function (Integrated)
# =====================================================================================

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss, integrated directly into main.py.
    
    This is a generalized implementation that works for both binary
    and multi-class classification.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): A tensor of weights for each class. 
                                      e.g., [0.25, 0.75] for binary.
                                      This is your class-weighting part!
            gamma (float, optional): The focusing parameter. Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # For NLLLoss (multi-class)
        if isinstance(alpha, (list, np.ndarray, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): The raw logits from the model (e.g., shape [N, C]).
            targets (Tensor): The ground truth labels (e.g., shape [N]).
        """
        
        # --- Handle Multi-Class (e.g., NLLLoss) ---
        if inputs.dim() > 1 and targets.dim() == 1:
            # targets shape [N] -> [N, 1]
            targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
            
            # Calculate log_softmax and probabilities
            log_pt = F.log_softmax(inputs, dim=1)
            pt = torch.exp(log_pt)
            
            # Gather the prob of the true class
            pt = (pt * targets_one_hot).sum(dim=1)
            log_pt = (log_pt * targets_one_hot).sum(dim=1)
            
            # Move alpha to device if needed
            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                
                # Select alpha for each sample based on its target class
                alpha_t = self.alpha.gather(0, targets)
            
        # --- Handle Binary / Multi-Label (e.g., BCEWithLogitsLoss) ---
        else:
            # For BCEWithLogitsLoss, targets should be float
            if targets.dtype != inputs.dtype:
                targets = targets.float()

            # Calculate probabilities
            pt = torch.sigmoid(inputs)
            
            # Calculate BCE loss manually
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            
            # This is p_t: prob of the *true* class
            # If target is 1, p_t = p. If target is 0, p_t = 1-p.
            pt = torch.where(targets == 1, pt, 1 - pt)
            
            # The "log_pt" in this case is -bce_loss
            log_pt = -bce_loss

            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                
                # For binary, alpha is often [alpha_0, alpha_1]
                # alpha_t will be alpha_1 for target=1, alpha_0 for target=0
                alpha_t = torch.where(targets == 1, self.alpha[1], self.alpha[0])

        # --- The Core Focal Loss Calculation ---
        modulating_factor = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            focal_loss = -alpha_t * modulating_factor * log_pt
        else:
            focal_loss = -modulating_factor * log_pt

        # --- Apply Reduction ---
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =====================================================================================
#  Section 2: Helper Functions
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

def save_results(model, tsne_labels, logits, labels, args, out_dir):
    """Saves all run artifacts to a directory."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'tsne_labels.npy'), tsne_labels)
    np.save(os.path.join(out_dir, 'logits.npy'), logits.cpu().detach().numpy())
    np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().detach().numpy())

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
    params.append(f"bs-{args.batch_size}")
    
    params.append(f"loss-{args.loss_fn}")
    if args.loss_fn == 'focal':
        params.append(f"gamma-{args.gamma}")

    # --- Model-Specific Hyperparameters ---
    gnn_type = args.gnn
    
    if gnn_type in ['gcn', 'gat', 'sage', 'gin']:
        params.append(f"layers-{args.local_layers}")
        if gnn_type == 'gat':
            params.append(f"heads-{args.num_heads}")
        if args.res: params.append('res')
        if args.ln: params.append('ln')
        if args.bn: params.append('bn')
        if args.jk: params.append('jk')
    
    return "_".join(params)

# =====================================================================================
#  Section 3: Main Training and Evaluation Function
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
    
    n, c, d = dataset.graph['num_nodes'], max(dataset.label.max().item() + 1, dataset.label.shape[1]), dataset.graph['node_feat'].shape[1]
    print(f'Dataset: {args.dataset}, Nodes: {n}, Classes: {c}, Features: {d}')
    
    # --- Create a unified PyG Data object for NeighborLoader ---
    print("--- Preparing PyG Data object for NeighborLoader ---")
    edge_index_full = to_undirected(dataset.graph['edge_index'])
    edge_index_full, _ = remove_self_loops(edge_index_full)
    edge_index_full, _ = add_self_loops(edge_index_full, num_nodes=n)
    
    pyg_data = Data(
        x=dataset.graph['node_feat'].cpu(),
        edge_index=edge_index_full.cpu(),
        y=dataset.label.cpu()
    )
    print(pyg_data)
    
    # --- Simplified Model Initialization ---
    print("--- Initializing Standard GNN Model ---")
    model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
                  heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
                  res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn=args.gnn).to(device)

    eval_func = {'prauc': eval_prauc, 'rocauc': eval_rocauc, 'balacc': eval_balanced_acc}.get(args.metric, eval_acc)
    logger = Logger(args.runs, args)
    
    all_acc, vals, all_balanced_acc, all_roc_auc, all_pr_auc = [], [], [], [], []
    all_reports_default, all_reports_optimal = [], []

    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'] # Keep on CPU
        model.reset_parameters()
        
        # --- Loss Initialization based on args.loss_fn ---
        class_weights = None
        if args.loss_fn in ['weighted_nll', 'focal']:
            print(f"Run {run+1}: Calculating class weights...")
            is_binary = args.dataset in ('questions') # or c == 2
            
            labels_train = pyg_data.y.squeeze(1)[train_idx]
            num_classes_for_weights = 2 if is_binary else c
            
            class_counts = torch.bincount(labels_train, minlength=num_classes_for_weights).float()
            print(f"  Class counts (train): {class_counts}")
            
            class_weights = 1. / (class_counts + 1e-8) # Inverse frequency
            class_weights = class_weights / class_weights.sum() # Normalize
            
            if not is_binary:
                # For NLLLoss, weights are scaled
                class_weights = class_weights * num_classes_for_weights 
                
            class_weights = class_weights.to(device)
            print(f"  Class weights: {class_weights}")

        # Initialize criterion
        if args.dataset in ('questions'): # Binary/Multi-label case
            if args.loss_fn == 'focal':
                print(f"Run {run+1}: Using FocalLoss (Binary) with alpha={class_weights} gamma={args.gamma}")
                criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
            elif args.loss_fn == 'weighted_nll':
                # Convert [weight_0, weight_1] to pos_weight
                pos_weight = class_weights[1] / (class_weights[0] + 1e-8)
                print(f"Run {run+1}: Using BCEWithLogitsLoss with pos_weight={pos_weight}")
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
            else: # 'nll'
                print(f"Run {run+1}: Using standard BCEWithLogitsLoss")
                criterion = nn.BCEWithLogitsLoss()
        
        else: # Multi-class case
            if args.loss_fn == 'focal':
                print(f"Run {run+1}: Using FocalLoss (Multi-class) with alpha={class_weights} gamma={args.gamma}")
                criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
            elif args.loss_fn == 'weighted_nll':
                print(f"Run {run+1}: Using NLLLoss with weights={class_weights}")
                criterion = nn.NLLLoss(weight=class_weights.to(device))
            else: # 'nll'
                print(f"Run {run+1}: Using standard NLLLoss")
                criterion = nn.NLLLoss()
        # --- END LOSS SECTION ---
        
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        # --- Define Neighbor Sampling Loader ---
        num_sampled_neighbors = [15, 10]
        if len(num_sampled_neighbors) != args.local_layers:
            num_sampled_neighbors = [10] * args.local_layers

        train_loader = NeighborLoader(
            pyg_data,
            num_neighbors=num_sampled_neighbors,
            batch_size=args.batch_size,
            input_nodes=train_idx,
            shuffle=True,
            num_workers=args.num_workers
        )

        best_val, best_test = float('-inf'), float('-inf')
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        for epoch in range(args.epochs):
            model.train()
            
            # --- Training Loop over Batches ---
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index)
                
                # --- Loss calculation (handles FocalLoss inputs) ---
                if args.dataset in ('questions'):
                    true_label_batch = F.one_hot(batch.y, batch.y.max() + 1).squeeze(1) if batch.y.shape[1] == 1 else batch.y
                    true_label_batch = true_label_batch.squeeze(1)[:batch.batch_size].to(torch.float)
                    
                    # FocalLoss (and BCE) expects logits
                    loss = criterion(out[:batch.batch_size], true_label_batch)
                
                else: # Multi-class
                    true_label_batch = batch.y.squeeze(1)[:batch.batch_size]
                    
                    if args.loss_fn == 'focal':
                        # FocalLoss expects logits
                        loss = criterion(out[:batch.batch_size], true_label_batch)
                    else:
                        # NLLLoss expects log_softmax
                        out_log_softmax = F.log_softmax(out, dim=1)
                        loss = criterion(out_log_softmax[:batch.batch_size], true_label_batch)
                # --- END Loss ---

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            epoch_loss = total_loss / len(train_loader)
            # --- END Batch Loop ---
            
            # --- Inlined evaluation logic ---
            with torch.no_grad():
                model.eval()
                pyg_data_eval = pyg_data.to(device)
                
                out = model(pyg_data_eval.x, pyg_data_eval.edge_index)
                
                # --- Train loss calculation (handles FocalLoss inputs) ---
                if args.dataset in ('questions'):
                    true_label_eval = F.one_hot(pyg_data_eval.y, pyg_data_eval.y.max() + 1).squeeze(1) if pyg_data_eval.y.shape[1] == 1 else pyg_data_eval.y
                    true_label_eval = true_label_eval.squeeze(1)[split_idx['train']].to(torch.float)
                    train_loss = criterion(out[split_idx['train']], true_label_eval)
                else:
                    true_label_eval = pyg_data_eval.y.squeeze(1)[split_idx['train']]
                    if args.loss_fn == 'focal':
                        train_loss = criterion(out[split_idx['train']], true_label_eval)
                    else:
                        out_log_softmax = F.log_softmax(out, dim=1)
                        train_loss = criterion(out_log_softmax[split_idx['train']], true_label_eval)
                # --- END Train Loss ---

                train_res = eval_func(pyg_data_eval.y[split_idx['train']], out[split_idx['train']])
                valid_res = eval_func(pyg_data_eval.y[split_idx['valid']], out[split_idx['valid']])
                test_res = eval_func(pyg_data_eval.y[split_idx['test']], out[split_idx['test']])
                
                result = (train_res, valid_res, test_res, train_loss.item())
                
                del pyg_data_eval
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # --- END Evaluation ---
            
            # logger.add_result(run, result[:-1])
            # NEW (Corrected)
            logger.add_result(run, result)

            if result[1] > best_val:
                best_val, best_test = result[1], result[2]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, Loss: {epoch_loss:.4f}, Train: {100*result[0]:.2f}%, Valid: {100*result[1]:.2f}%, Test: {100*result[2]:.2f}%')

        print(f'Run {run+1}/{args.runs}: Best Valid: {100*best_val:.2f}%, Best Test: {100*best_test:.2f}%')

        # --- Final Evaluation ---
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            pyg_data_eval = pyg_data.to(device)
            out = model(pyg_data_eval.x, pyg_data_eval.edge_index)
            y_true, test_idx = pyg_data_eval.y, split_idx['test']

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

        # --- Embedding Extraction and Saving (Simplified) ---
        with torch.no_grad():
            labels_cpu = y_true[test_idx].cpu().numpy().squeeze()
            
        param_string = generate_param_string(args)
        
        # --- MODIFIED: Changed 'results' to 'results_batching' ---
        out_dir = f'results_batching/{args.dataset}/{args.gnn}/{param_string}/run_{run}'
        
        print('save results started')
        save_results(model, labels_cpu, out[split_idx['test']], y_true[test_idx], args, out_dir)
        # plot_tsne(tsne_embeds, labels_cpu, out_dir) # t-SNE requires embeddings
        plot_logits_distribution(out[split_idx['test']], y_true[test_idx], out_dir)
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
    
    param_string = generate_param_string(args) # Or re-use from the last run
    
    # --- MODIFIED: Changed 'results' to 'results_batching' ---
    summary_dir = f'results_batching/{args.dataset}/{args.gnn}/{param_string}/summary'
    
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(summary_dir, 'reports_default.json'), 'w') as f:
        json.dump(all_reports_default, f, indent=2)
    if all_reports_optimal:
        with open(os.path.join(summary_dir, 'reports_optimal.json'), 'w') as f:
            json.dump(all_reports_optimal, f, indent=2)

# =====================================================================================
#  Section 4: Script Entry Point
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrated GNN Training Pipeline')
    parser_add_main_args(parser)
    
    # --- Arguments for Batching and Loss ---
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading.')
    
    parser.add_argument('--loss_fn', type=str, default='nll',
                        choices=['nll', 'weighted_nll', 'focal'],
                        help='Loss function to use.')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss.')
    # --- END Arguments ---

    args = parser.parse_args()
    
    if args.gnn in ['fsgcn', 'glognn', 'gprgnn', 'mlp']:
        print(f"Error: GNN type '{args.gnn}' is no longer supported in this script.")
        print("Please use a standard GNN like 'gcn', 'gat', 'sage', or 'gin'.")
        sys.exit(1)
        
    print(args)
    train_and_evaluate(args)