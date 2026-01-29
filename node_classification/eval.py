


# eval.py

import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None, 
             fsgcn_list_mat=None, 
             glognn_features=None, glognn_adj_sparse=None, glognn_adj_dense=None,
             gprgnn_features=None):
    """
    Evaluates the model on the given data splits.
    Handles standard GNNs, FSGCN, GloGNN, and GPRGNN.
    """
    if result is not None:
        out = result
    else:
        model.eval()
        # Conditional forward pass based on the GNN type
        if args.gnn == 'fsgcn':
            out = model(fsgcn_list_mat)
        elif args.gnn == 'glognn':
            out = model(glognn_features, glognn_adj_sparse, glognn_adj_dense)
        elif args.gnn == 'gprgnn':
            out = model(gprgnn_features, dataset.graph['edge_index'])
        elif args.gnn == 'mlp':
            out = model(dataset.graph['node_feat'])
        else: # Standard GNN
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        # Note: log_softmax is applied to the output for NLLLoss
        out_log_softmax = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out_log_softmax[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None, 
                 fsgcn_list_mat=None, glognn_features=None, glognn_adj_sparse=None, glognn_adj_dense=None, 
                 gprgnn_features=None):
    """
    Evaluates the model on the CPU.
    Handles all GNN types.
    """
    if result is not None:
        out = result
    else:
        model.eval()

    # Move all necessary components to CPU
    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    
    if args.gnn == 'fsgcn':
        fsgcn_list_mat_cpu = [mat.to(torch.device("cpu")) for mat in fsgcn_list_mat]
        out = model(fsgcn_list_mat_cpu)
    elif args.gnn == 'glognn':
        glognn_features_cpu = glognn_features.cpu()
        glognn_adj_sparse_cpu = glognn_adj_sparse.cpu()
        glognn_adj_dense_cpu = glognn_adj_dense.cpu()
        out = model(glognn_features_cpu, glognn_adj_sparse_cpu, glognn_adj_dense_cpu)
    elif args.gnn == 'gprgnn':
        gprgnn_features_cpu = gprgnn_features.cpu()
        edge_index_cpu = dataset.graph['edge_index'].cpu()
        out = model(gprgnn_features_cpu, edge_index_cpu)
    else:
        edge_index_cpu, x_cpu = dataset.graph['edge_index'].cpu(), dataset.graph['node_feat'].cpu()
        out = model(x_cpu, edge_index_cpu)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
        
    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out_log_softmax = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out_log_softmax[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
            
    # Move model back to the original device after evaluation
    model.to(device)
    dataset.label = dataset.label.to(device)

    return train_acc, valid_acc, test_acc, valid_loss, out
