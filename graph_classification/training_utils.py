# training_utils.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
# --- MODIFIED: Added torcheval imports ---
from torcheval.metrics.functional import binary_auprc, multiclass_auprc

def train(model, loader, optimizer, device, dataset_type):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss() if dataset_type == 'ogb' else torch.nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        if batch.x.shape[0] <= 1: continue
        optimizer.zero_grad()
        pred = model(batch)
        if dataset_type == 'ogb':
            is_labeled = batch.y == batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        else:
            loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate_graphclass(model, loader, device, dataset_type, evaluator=None):
    model.eval()
    y_true_all, y_pred_all = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        if batch.x.shape[0] <= 1: continue
        pred = model(batch)
        y_true_all.append(batch.y)
        y_pred_all.append(pred)

    y_true = torch.cat(y_true_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)
    
    # --- Comprehensive Metric Calculation ---
    if dataset_type == 'ogb':
        y_true_np = y_true.cpu().numpy()
        y_pred_binary_np = (y_pred.cpu().numpy() > 0).astype(int)
        
        primary_metric = evaluator.eval({"y_true": y_true, "y_pred": y_pred}) if evaluator else {'rocauc': 0}
        
        acc_list, bal_acc_list, pr_auc_list = [], [], []
        for i in range(y_true.shape[1]): # Iterate over tasks
            is_labeled = ~torch.isnan(y_true[:, i])
            if is_labeled.sum() > 0:
                y_true_task_np = y_true_np[is_labeled.cpu(), i]
                y_pred_binary_task_np = y_pred_binary_np[is_labeled.cpu(), i]

                acc_list.append(accuracy_score(y_true_task_np, y_pred_binary_task_np))
                bal_acc_list.append(balanced_accuracy_score(y_true_task_np, y_pred_binary_task_np))

                # --- MODIFIED: Use torcheval for AUPR on tensors ---
                y_true_task = y_true[is_labeled, i]
                y_pred_task = y_pred[is_labeled, i]
                
                # Check if both classes are present for the task
                if (y_true_task == 0).any() and (y_true_task == 1).any():
                    task_probs = torch.sigmoid(y_pred_task)
                    aupr = binary_auprc(input=task_probs, target=y_true_task.long())
                    pr_auc_list.append(aupr.item())

        metrics = {
            'acc': np.mean(acc_list) if acc_list else 0,
            'balacc': np.mean(bal_acc_list) if bal_acc_list else 0,
            'prauc': np.mean(pr_auc_list) if pr_auc_list else 0,
            'rocauc': primary_metric.get('rocauc', 0)
        }
    else: # TU Datasets
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.argmax(dim=1).cpu().numpy()
        
        metrics = {
            'acc': accuracy_score(y_true_np, y_pred_np),
            'balacc': balanced_accuracy_score(y_true_np, y_pred_np),
        }
        try:
            # --- MODIFIED: Use torcheval for AUPR and keep tensors ---
            if y_pred.shape[1] == 2:
                probs = F.softmax(y_pred, dim=1)
                metrics['rocauc'] = roc_auc_score(y_true_np, probs[:, 1].cpu().numpy())
                # Use torcheval's binary_auprc
                metrics['prauc'] = binary_auprc(input=probs[:, 1], target=y_true.squeeze().long()).item()
            else:
                probs = F.softmax(y_pred, dim=1)
                metrics['rocauc'] = roc_auc_score(y_true_np, probs.cpu().numpy(), multi_class='ovr')
                # Use torcheval's multiclass_auprc
                metrics['prauc'] = multiclass_auprc(
                    input=probs, target=y_true.squeeze().long(), 
                    num_classes=y_pred.shape[1], average='macro'
                ).item()
        except (ValueError, RuntimeError):
            metrics['rocauc'], metrics['prauc'] = 0.0, 0.0

    return y_true, y_pred, metrics