import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from GraphSAGE_model import MeanAggregator, Encoder, SupervisedGraphSage
from GCN_model import GCN
from GAT_model import GAT
from load_data import load_data
from load_data import load_data2
# import visdom
import matplotlib.pyplot as plt

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
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item(), acc_train.data.item(), acc_val.data.item()

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

def myplot(loss_values, acc_train_values, acc_val_values, model_str):
    plt.figure(figsize=(12, 8))
    x = list(np.linspace(0, 99, 100))
    # plt.plot(x, loss_values, label="{}:loss_values".format(model_str))
    plt.plot(x, acc_train_values, label="{}:acc_train_values".format(model_str))
    plt.plot(x, acc_val_values, label="{}:acc_val_values".format(model_str))
    # plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()
    # plt.savefig('./image/GraphSAGE6.jpg')
    plt.show()

def cora_train(Model):
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disables CUDA training.")
    parser.add_argument("--fastmode", action="store_true", default=False, help="Validate during training pass.") # 设置为True后训练速度会提升,val_acc会下降,但acc_test不受影响
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight dacay (L2 loss on parameters).")
    parser.add_argument("--hidden", type=int, default=8, help="Number of hidden units.")
    parser.add_argument("--nb_heads", type=int, default=8, help="Number of head attentions.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability).")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
    parser.add_argument("--patience", type=int, default=100, help="Patience")
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)  # 导致训练效果大幅下降
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    # _, labels2, _ = load_data2()
    # print(labels)
    # print(labels[1])
    # print("-"*50)
    # print(labels2)
    # print(labels2[1])
    if Model == SupervisedGraphSage:
        agg1 = MeanAggregator()
        enc1 = Encoder(feature_dim=1433, embed_dim=128,
                       aggregator=agg1)  # gcn=False
        agg2 = MeanAggregator()
        enc2 = Encoder(feature_dim=enc1.embed_dim, embed_dim=128,
                       aggregator=agg2, base_model=enc1)  # gcn=False
        # agg3 = MeanAggregator()
        # enc3 = Encoder(feature_dim=enc2.embed_dim, embed_dim=128,
        #                aggregator=agg3, base_model=enc2)  # gcn=False
        model = SupervisedGraphSage(num_classes=7, dropout=args.dropout, enc=enc2)
    elif Model == GCN:
        model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
    elif Model == GAT:
        model = GAT(nfeat=features.shape[1], nhid=args.hidden, nclass=int(labels.max()) + 1, dropout=args.dropout,
                    nheads=args.nb_heads, alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # t_total = time.time()
    loss_values = []
    acc_train_values = []
    acc_val_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss, acc_train, acc_val = train(epoch=epoch, model=model, optimizer=optimizer, features=features, adj=adj, labels=labels,
              idx_train=idx_train, idx_val=idx_val, fastmode=args.fastmode)
        loss_values.append(loss)
        acc_train_values.append(acc_train)
        acc_val_values.append(acc_val)
        if Model == SupervisedGraphSage:
            torch.save(model.state_dict(), "GraphSAGE.{}.pkl".format(epoch))
        elif Model == GCN:
            torch.save(model.state_dict(), "GCN.{}.pkl".format(epoch))
        elif Model == GAT:
            torch.save(model.state_dict(), "GAT.{}.pkl".format(epoch))
        # x = np.linspace(0, epoch, epoch + 1)
        # vis.line(X=x, Y=loss_values, win="1", opts=dict(title='{}:loss_values'.format(str)))
        # vis.line(X=x, Y=acc_train_values, win="2", opts=dict(title='{}:acc_train_values'.format(str)))
        # vis.line(X=x, Y=acc_val_values, win="3", opts=dict(title='{}:acc_val_values'.format(str)))
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
    myplot(loss_values, acc_train_values, acc_val_values, model_str="GraphSAGE")
    test(model=model, features=features, adj=adj, labels=labels, idx_test=idx_test)

if __name__ =="__main__":
    # vis = visdom.Visdom()
    # vis = visdom.Visdom(port=6006)

    # cora_train(SupervisedGraphSage)
    # cora_train(GCN)
    cora_train(GAT)