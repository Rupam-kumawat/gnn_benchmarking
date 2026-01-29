
import torch
import torch.nn.functional as F
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, balanced_accuracy_score
from torcheval.metrics.functional import binary_auprc, multiclass_auprc



def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """randomly splits label into train/valid/test splits"""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    """use all remaining data points as test data, so test_num will not be used"""
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num : valid_num + test_num],
    )
    print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    return split_idx


def load_fixed_splits(data_dir, dataset, name):
    splits_lst = []
    if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            splits_lst.append(splits)
    elif name in ['wikics']:
        torch_dataset = WikiCS(root=f"{data_dir}/wikics/")
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(torch.logical_or(data.val_mask, data.stopping_mask)[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:])[0]
            splits_lst.append(splits)
    elif name in ['amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics']:
        splits = {}
        idx = np.load(f'{data_dir}/{name}_split.npz')
        splits['train'] = torch.from_numpy(idx['train'])
        splits['valid'] = torch.from_numpy(idx['valid'])
        splits['test'] = torch.from_numpy(idx['test'])
        splits_lst.append(splits)
    elif name in ['pokec']:
        split = np.load(f'{data_dir}/{name}/{name}-splits.npy', allow_pickle=True)
        for i in range(split.shape[0]):
            splits = {}
            splits['train'] = torch.from_numpy(np.asarray(split[i]['train']))
            splits['valid'] = torch.from_numpy(np.asarray(split[i]['valid']))
            splits['test'] = torch.from_numpy(np.asarray(split[i]['test']))
            splits_lst.append(splits)
    elif name in ["chameleon", "squirrel", "squirrel-filtered"]:
        file_path = f"{data_dir}/geom-gcn/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N), 10 splits
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)
            
    elif name in ['wiki-cooc']:
        file_path = f"{data_dir}/wiki_cooc.npz"
        data = np.load(file_path, allow_pickle=True)
        train_masks = data["train_masks"]  # (num_splits, N)
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        num_splits = train_masks.shape[0]
        print('num_splits', num_splits)

        for i in range(num_splits):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"]  = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)

    else:
        raise NotImplementedError

    return splits_lst


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list)/len(acc_list)

def eval_acc(y_true, y_pred):
    """
    Calculates standard accuracy for binary or multi-class classification.
    """
    # Detach tensors and get predictions
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    # Squeeze labels to be 1D
    y_true = y_true.squeeze()

    # Filter out unlabeled nodes (assumed to be NaN)
    is_labeled = ~np.isnan(y_true)
    y_true = y_true[is_labeled]
    y_pred = y_pred[is_labeled]
    
    # Calculate accuracy
    correct = y_true == y_pred
    return float(np.sum(correct)) / len(correct) if len(correct) > 0 else 0.0

def eval_balanced_acc(y_true, y_pred):
    """
    Calculates balanced accuracy for binary or multi-class classification.
    This is especially useful for imbalanced datasets.
    """
    # Detach tensors and get predictions
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
    
    # Squeeze labels to be 1D
    y_true = y_true.squeeze()

    # Filter out unlabeled nodes
    is_labeled = ~np.isnan(y_true)
    y_true = y_true[is_labeled]
    y_pred = y_pred[is_labeled]

    if len(y_true) == 0:
        return 0.0
        
    # balanced_accuracy_score handles both binary and multi-class cases
    return balanced_accuracy_score(y_true, y_pred)

def eval_rocauc(y_true, y_pred):
    """
    Calculates ROC-AUC score for binary or multi-class classification.
    """
    # Detach tensors from the computation graph
    y_true = y_true.detach()
    y_pred = y_pred.detach()

    # Squeeze labels to be 1D
    y_true_np = y_true.squeeze().cpu().numpy()

    # Convert model logits to probabilities
    y_pred_probs = F.softmax(y_pred, dim=-1).cpu().numpy()

    # Filter out any potential unlabeled data
    is_labeled = ~np.isnan(y_true_np)
    y_true_np = y_true_np[is_labeled]
    y_pred_probs = y_pred_probs[is_labeled]
    
    if len(y_true_np) == 0:
        return 0.0

    num_classes = y_pred.shape[1]

    if num_classes == 2:
        # For binary classification, roc_auc_score expects the probability of the positive class
        return roc_auc_score(y_true_np, y_pred_probs[:, 1])
    else:
        # For multi-class, use the One-vs-Rest strategy
        return roc_auc_score(y_true_np, y_pred_probs, multi_class='ovr')
    
    
def eval_aupr_binary(y_true, y_pred):
    """
    Calculates the Area Under the Precision-Recall Curve (AUPR) for BINARY
    classification using the torcheval library.

    Args:
        y_true (torch.Tensor): Ground truth labels. Shape: (N, 1) or (N,).
        y_pred (torch.Tensor): Predicted logits or scores. Shape: (N, 2).
    """
    # Ensure tensors are detached from the computation graph
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    
    # Flatten y_true to a 1D tensor of shape (N,) and ensure it's long type
    y_true_flat = y_true.squeeze().long()
    
    # Get probabilities for the positive class (class 1)
    # The 'input' for binary_auprc should be a 1D tensor of scores for the positive class
    y_pred_probs = F.softmax(y_pred, dim=-1)[:, 1]
    
    # Filter out any potential unlabeled data (represented as NaNs)
    is_labeled = ~torch.isnan(y_true_flat)
    
    # Check if both positive and negative classes are present, as AUPRC is only defined in this case
    if torch.sum(y_true_flat[is_labeled] == 1) > 0 and torch.sum(y_true_flat[is_labeled] == 0) > 0:
        # Call the torcheval function
        # Note: The argument is 'input', not 'preds', and there is no 'reorder' parameter.
        score = binary_auprc(
            input=y_pred_probs[is_labeled],
            target=y_true_flat[is_labeled]
        )
        return score.item()
    else:
        # If a class is missing, AUPRC is not well-defined.
        raise RuntimeError('Cannot compute AUPR. Both positive and negative samples are required.')

def eval_aupr_multiclass(y_true, y_pred):
    """
    Calculates the macro-average AUPR for MULTI-CLASS classification 
    using the torcheval library.

    Args:
        y_true (torch.Tensor): Ground truth labels (class indices). Shape: (N, 1) or (N,).
        y_pred (torch.Tensor): Predicted logits or scores. Shape: (N, num_classes).
    """
    # Ensure tensors are detached
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    
    num_classes = y_pred.shape[1]
    
    # Flatten y_true to a 1D tensor of shape (N,) and ensure it's long type
    y_true_flat = y_true.squeeze().long()
    
    # Convert logits to probabilities, as required by the 'input' parameter
    y_pred_probs = F.softmax(y_pred, dim=-1)
    
    # Filter out any potential unlabeled data
    is_labeled = ~torch.isnan(y_true_flat)
    
    # Call the torcheval function
    # Note: No 'reorder' parameter is needed.
    score = multiclass_auprc(
        input=y_pred_probs[is_labeled],
        target=y_true_flat[is_labeled],
        num_classes=num_classes,
        average="macro" # "macro" gives equal weight to each class
    )
    return score.item()

def eval_prauc(y_true, y_pred):
    """
    A wrapper function that automatically selects the binary or multi-class AUPR
    calculation based on the shape of the prediction tensor.
    """
    # Infer the number of classes from the model's output shape
    num_classes = y_pred.shape[1]
    
    # A binary task can sometimes have an output shape of 1, but usually 2 for CE loss.
    if num_classes <= 2:
        # print("Detected binary classification task.")
        return eval_aupr_binary(y_true, y_pred)
    else:
        # print(f"Detected multi-class classification task with {num_classes} classes.")
        return eval_aupr_multiclass(y_true, y_pred)


dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}



