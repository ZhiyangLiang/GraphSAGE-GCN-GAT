import time
import random
import numpy as np
import torch
import torch.nn as nn
from GraphSAGE_model import MeanAggregator
from GraphSAGE_model import SumAggregator
from GraphSAGE_model import MaxAggregator
from GraphSAGE_model import Encoder
from GraphSAGE_model import SupervisedGraphSage
from GraphSAGE_model import GCNAggregator
from load_data import load_cora

def train(nodes, labels, model, optimizer):
    times = []
    losses = []
    for i in range(100):
        start_time = time.time()
        optimizer.zero_grad()
        loss = model.loss(nodes, labels)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)

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
    np.random.seed(1)  # 应对用np.random生成的随机数
    random.seed(1)  # 应对用random生成的随机数
    num_nodes = 2708
    num_feats = 1433
    feat_data, labels, adj_lists = load_cora()
    labels = torch.tensor(labels)  # torch.LongTensor等价于torch.tensor
    features = nn.Embedding(num_nodes, num_feats)  # torch.FloatTensor等价于torch.Tensor
    features.weight = nn.Parameter(torch.FloatTensor(feat_data),
                                   requires_grad=False)
    # 上一行的作用：加快训练速度，提升训练效果(F1)
    # print(features)
    # print(features.weight.sum(dim=1))

    agg1 = GCNAggregator(features)
    enc1 = Encoder(features=features, feature_dim=1433, embed_dim=128, adj_lists=adj_lists,
                   aggregator=agg1)  # gcn=False
    agg2 = MeanAggregator(features=lambda nodes: enc1(nodes).t())
    enc2 = Encoder(features=lambda nodes: enc1(nodes).t(), feature_dim=enc1.embed_dim, embed_dim=128,
                   adj_lists=adj_lists, aggregator=agg2, base_model=enc1)  # gcn=False
    enc1.num_sample = 5
    enc2.num_sample = 5

    graphsage = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    # test = rand_indices[:1000]
    # val = rand_indices[1000:1500]
    # train = list(rand_indices[1500:])
    test = rand_indices[:140]
    val = rand_indices[200:500]
    train = list(rand_indices[500:1500])
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              torch.LongTensor(labels[np.array(batch_nodes)].reshape(1, -1)[0]))  # Variable
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        val_output = graphsage.forward(val)
        acc_val = accuracy(val_output, labels[val])
        print("epoch:{}".format(batch + 1), "loss:{}".format(loss.item()), "accuracy:{}".format(acc_val))

if __name__ =="__main__":
    cora_train()