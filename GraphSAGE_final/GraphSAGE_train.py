import os
import glob
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphSAGE_model import MeanAggregator
from GraphSAGE_model import SumAggregator
from GraphSAGE_model import MaxAggregator
from GraphSAGE_model import Encoder
from GraphSAGE_model import SupervisedGraphSage
from GraphSAGE_model import GCNAggregator
from GraphSAGE_model import GATAggregator
from load_data import load_cora

def train(epoch, nodes, val_nodes, labels, model, optimizer):
    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    loss_train = model.loss(nodes, torch.LongTensor(labels[np.array(nodes)].reshape(1, -1)[0]))
    output = model(nodes)
    acc_train = accuracy(output, labels[nodes])
    loss_train.backward()
    optimizer.step()

    model.eval()
    loss_val = model.loss(val_nodes, torch.LongTensor(labels[np.array(val_nodes)].reshape(1, -1)[0]))
    output = model(val_nodes)
    acc_val = accuracy(output, labels[val_nodes])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - start_time))
    return loss_val.item(), acc_train.data.item(), acc_val.data.item()

def test(model, test_nodes, labels):
    model.eval()
    output = model(test_nodes)
    loss_test = model.loss(test_nodes, torch.LongTensor(labels[np.array(test_nodes)].reshape(1, -1)[0]))
    acc_test = accuracy(output, labels[test_nodes])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

def predict(nodes, labels, model):
    after_model = model(nodes)
    predicted = torch.argmax(after_model, dim=1)
    acc = (torch.sum(predicted == labels) / len(labels)).item()
    print(acc)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels).reshape(labels.shape)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def cora_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disables CUDA training.")
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

    num_nodes = 2708
    num_feats = 1433
    feat_data, labels, adj_lists = load_cora()
    labels = torch.LongTensor(labels)  # torch.LongTensor等价于torch.tensor
    features = nn.Embedding(num_nodes, num_feats)  # torch.FloatTensor等价于torch.Tensor
    features.weight = nn.Parameter(torch.FloatTensor(feat_data),
                                   requires_grad=False)

    # agg1 = GATAggregator(features, features_dim=1433)
    agg1 = MeanAggregator(features)
    enc1 = Encoder(features=features, feature_dim=1433, embed_dim=128, adj_lists=adj_lists,
                   aggregator=agg1)  # gcn=False
    agg2 = MeanAggregator(features=lambda nodes: enc1(nodes).t())
    enc2 = Encoder(features=lambda nodes: enc1(nodes).t(), feature_dim=enc1.embed_dim, embed_dim=128,
                   adj_lists=adj_lists, aggregator=agg2, base_model=enc1)  # gcn=False
    enc1.num_sample = 5
    enc2.num_sample = 5

    model = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    idx_test = rand_indices[:1000]
    idx_val = rand_indices[1000:1500]
    idx_train = rand_indices[1500:]
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.7)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_values = []
    acc_train_values = []
    acc_val_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        batch_nodes = idx_train[:256]
        random.shuffle(idx_train)

        loss, acc_train, acc_val = train(epoch=epoch, nodes=batch_nodes, val_nodes=idx_val, labels=labels, model=model, optimizer=optimizer)
        loss_values.append(loss)
        acc_train_values.append(acc_train)
        acc_val_values.append(acc_val)
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
    test(test_nodes=idx_test, labels=labels, model=model)

if __name__ =="__main__":
    cora_train()