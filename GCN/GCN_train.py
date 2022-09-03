import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from GCN_model import GCN
from load_data import load_data

def train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val, fastmode):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        model.eval()  # 注意：dropout会影响前向传播,从而影响预测结果
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()

def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels).reshape(labels.shape)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def cora_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disables CUDA training.")
    parser.add_argument("--fastmode", action="store_true", default=True, help="Validate during training pass.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay(L2 loss on parameters).")
    parser.add_argument("--hidden", type=int, default=8, help="Number of hidden units.")
    parser.add_argument("--nb_heads", type=int, default=8, help="Number of head attentions.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability).")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
    parser.add_argument("--patience", type=int, default=100, help="Patience")
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch=epoch, model=model, optimizer=optimizer, features=features, adj=adj, labels=labels, idx_train=idx_train, idx_val=idx_val, fastmode=args.fastmode))
        torch.save(model.state_dict(), "GCN.{}.pkl".format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break

        files = glob.glob("*.pkl")
        for file in files:
            epoch_nb = int(file.split(".")[1])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[1])
        if epoch_nb > best_epoch:
            os.remove(file)

    test(model=model, features=features, adj=adj, labels=labels, idx_test=idx_test)

if __name__ =="__main__":
    cora_train()