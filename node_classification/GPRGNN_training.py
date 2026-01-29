import torch.nn.functional as F
import torch
def RunExp(args, Net, x, edge_index, labels_th, train_nodes, valid_nodes, test_nodes, device, loss_fn, metric):
    # 
    def train(model, optimizer, x, edge_index, labels_th, train_nodes, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)[train_nodes]
        loss = loss_fn(out, labels_th[train_nodes])
        loss.backward()
        optimizer.step()
        del out
    #
    def test(model, x, edge_index, train_nodes, valid_nodes, test_nodes):
        model.eval()
        logits, accs, losses = model(x, edge_index), [], []
        for mask in [train_nodes, valid_nodes, test_nodes]:
            # pred = logits[mask].max(1)[1]
            # acc = pred.eq(labels_th[mask]).sum().item() / len(mask)
            acc = metric(logits[mask], labels_th[mask]).item()
            loss = loss_fn(model(x, edge_index)[mask], labels_th[mask])
            # preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu().item())
        return accs, losses
    #
    appnp_net = Net
    model = appnp_net.to(device)
    optimizer = torch.optim.Adam([{
        'params': model.lin1.parameters(),
        'weight_decay': args.weight_decay, 'lr': args.lr
    },
        {
        'params': model.lin2.parameters(),
        'weight_decay': args.weight_decay, 'lr': args.lr
    },
        {
        'params': model.prop1.parameters(),
        'weight_decay': 0.0, 'lr': args.lr
    }], lr=args.lr)
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    for epoch in range(args.epoch_num):
        # 
        train(model, optimizer, x, edge_index, labels_th, train_nodes, args.dprate)
        # 
        [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss] = test(
            model, x, edge_index, train_nodes, valid_nodes, test_nodes)
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            TEST = appnp_net.prop1.temp.clone()
            Alpha = TEST.detach().cpu().numpy()
            Gamma_0 = Alpha
        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, Gamma_0
