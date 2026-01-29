import torch.nn.functional as F
from copy import deepcopy

import torch

# from sklearn.metrics import roc_auc_score

@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean().item()

# @torch.no_grad()
# def roc_auc(pr_logits, gt_labels):
#     return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy())

__all__ = ['train_step', 'val_step']


def train_step(
        model,
        optimizer,
        labels,
        list_mat,
        mask,
        loss_fn,
        metric,
        device: str = 'cpu'
):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat).squeeze()
    loss_train = loss_fn(output[mask], labels[mask].to(device))
    acc_train = metric(output[mask], labels[mask].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train, acc_train


def val_step(
        model,
        labels,
        list_mat,
        mask,
        loss_fn,
        metric,
        device: str = 'cpu'
):
    model.eval()
    with torch.no_grad():
        output = model(list_mat).squeeze()
        loss_val = loss_fn(output[mask], labels[mask].to(device))
        acc_val = metric(output[mask], labels[mask].to(device))
        return loss_val, acc_val
    

def run_on_split(
        model,
        optimizer,
        features,
        labels,
        list_mat,
        train_mask,
        val_mask,
        test_mask,
        device,
        args,
        loss_fn=F.cross_entropy,
        metric=accuracy
):
    # metric = accuracy # if len(torch.unique(labels)) > 2 else roc_auc
    best = -torch.inf
    best_params = None
    bad_counter = 0
    for step in range(args.epoch_num):
        loss_train, metric_train = train_step(
            model, optimizer, labels, list_mat, train_mask, loss_fn, metric, device=device)
        loss_val, metric_val = val_step(
            model, labels, list_mat, val_mask, loss_fn, metric, device=device)
        # if step % args.log_freq == 0:
        if step % 100 == 0:
            print(f'Train metric {metric_train:.4f} / Val acc {metric_val:.4f}')
        if metric_val > best:
            best = metric_val
            bad_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break
    # load best params
    model.load_state_dict(best_params)
    loss_test, metric_test = val_step(
        model, labels, list_mat, test_mask, loss_fn, metric, device=device)
    # return test accuracy
    return metric_test
