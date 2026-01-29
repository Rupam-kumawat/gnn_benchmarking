# import torch
# import torch.nn.functional as F

# @torch.no_grad()
# def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
#     if result is not None:
#         out = result
#     else:
#         model.eval()
#         if args.model == "nagphormer":
#             out = model(dataset.graph['node_feat'])
#         else:
#             out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
#     if args.model == "nodeformer":
#         out, loss_add = out
#     train_acc = eval_func(
#         dataset.label[split_idx['train']], out[split_idx['train']])
#     valid_acc = eval_func(
#         dataset.label[split_idx['valid']], out[split_idx['valid']])
#     test_acc = eval_func(
#         dataset.label[split_idx['test']], out[split_idx['test']])

#     if args.dataset in ('questions'):
#         if dataset.label.shape[1] == 1:
#             true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
#         else:
#             true_label = dataset.label
#         valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
#             split_idx['valid']].to(torch.float))
#     else:
#         out = F.log_softmax(out, dim=1)
#         valid_loss = criterion(
#             out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

#     return train_acc, valid_acc, test_acc, valid_loss, out

# @torch.no_grad()
# def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
#     if result is not None:
#         out = result
#     else:
#         model.eval()

#     model.to(torch.device("cpu"))
#     dataset.label = dataset.label.to(torch.device("cpu"))
#     edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
#     out = model(x, edge_index)

#     train_acc = eval_func(
#         dataset.label[split_idx['train']], out[split_idx['train']])
#     valid_acc = eval_func(
#         dataset.label[split_idx['valid']], out[split_idx['valid']])
#     test_acc = eval_func(
#         dataset.label[split_idx['test']], out[split_idx['test']])
#     if args.dataset in ('questions'):
#         if dataset.label.shape[1] == 1:
#             true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
#         else:
#             true_label = dataset.label
#         valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
#             split_idx['valid']].to(torch.float))
#     else:
#         out = F.log_softmax(out, dim=1)
#         valid_loss = criterion(
#             out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

#     return train_acc, valid_acc, test_acc, valid_loss, out



# This is the new, simplified eval.py for your NodeFormer project.
# It is identical to the GNN's eval.py because our adapted model
# now has the same simple interface.

import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    """
    Evaluates the model on the given dataset splits.
    This function works seamlessly with NodeFormerAdapted because it expects
    the same inputs and provides the same outputs as a standard GNN.
    """
    if result is not None:
        out = result
    else:
        model.eval()
        # A simple, direct forward pass now works for our adapted model.
        # No more special conditions are needed.
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    # The rest of the evaluation logic is standard.
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    # The loss calculation in the GNN script was already correct.
    # We don't apply log_softmax here if it's already done in the main loop's
    # final evaluation, but for the validation loss during training, this is correct.
    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        # Note: The main training loop often applies log_softmax before calling this.
        # This ensures validation loss is calculated correctly.
        temp_out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            temp_out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    # The 'out' returned here is the raw logits, which is what the main loop expects.
    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
    """
    A version of evaluate that forces execution on the CPU.
    """
    if result is not None:
        out = result
    else:
        model.eval()

    # Move model and data to CPU for evaluation
    cpu_device = torch.device("cpu")
    model.to(cpu_device)
    
    edge_index = dataset.graph['edge_index'].to(cpu_device)
    x = dataset.graph['node_feat'].to(cpu_device)
    cpu_label = dataset.label.to(cpu_device)
    
    out = model(x, edge_index)

    train_acc = eval_func(
        cpu_label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        cpu_label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        cpu_label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('questions'):
        if cpu_label.shape[1] == 1:
            true_label = F.one_hot(cpu_label, cpu_label.max() + 1).squeeze(1)
        else:
            true_label = cpu_label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        temp_out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            temp_out[split_idx['valid']], cpu_label.squeeze(1)[split_idx['valid']])
    
    # Move model and original data back to the original device
    model.to(device)

    return train_acc, valid_acc, test_acc, valid_loss, out