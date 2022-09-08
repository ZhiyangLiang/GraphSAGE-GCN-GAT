import time
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import Linear
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from collections import defaultdict
from torch_geometric.datasets import KarateClub
from GraphSAGE_model import MeanAggregator
from GraphSAGE_model import Encoder
from GraphSAGE_model import SupervisedGraphSage
from load_data import load_cora

def get_to_neighs(G):
    num_nodes = len(G.nodes)
    to_neighs = []
    for i in range(num_nodes):
        mid_set = set()
        for j in G.neighbors(i):
            mid_set.add(j)
        to_neighs.append(mid_set)
    return to_neighs


def create_elements(graph=nx.karate_club_graph(), num_embed=34, feature_dim=16, embed_dim=16, num_classes=4, nodes=None):
    if nodes == None:
        # print(1)
        nodes = []
        for i in range(len(graph.nodes)):
            nodes.append(i)
    # print(nodes)
    split = int(len(nodes)/3*2)
    features = nn.Embedding(num_embed, embed_dim)
    to_neighs = get_to_neighs(graph)
    # print(to_neighs)
    aggregator = MeanAggregator(features)
    enc = Encoder(features=features, feature_dim=feature_dim, embed_dim=embed_dim, adj_lists=to_neighs, aggregator=aggregator)
    model = SupervisedGraphSage(num_classes=num_classes, enc=enc)
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return nodes, to_neighs, model, optimizer, features, split


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
    # print(times)
    # print(losses)


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

# 此处训练集与测试集的划分为2:1,默认采用的数据集太小,故效果并不理想
def graph_train(dataset=KarateClub()):
    data = dataset[0]
    # print(data)
    labels = data.y
    nodes, to_neights, model, optimizer, features, split = create_elements(graph=nx.karate_club_graph(), num_embed=34,
                                                                           feature_dim=16, embed_dim=16, num_classes=4,
                                                                           nodes=None)
    train(nodes=nodes[:split], labels=labels[:split], model=model, optimizer=optimizer)
    predict(nodes=nodes[split:], labels=labels[split:], model=model)


def cora_train():
    np.random.seed(1)  # 应对用np.random生成的随机数
    random.seed(1)  # 应对用random生成的随机数
    num_nodes = 2708
    num_feats = 1433
    feat_data, labels, adj_lists = load_cora()
    labels = torch.tensor(labels)
    features = nn.Embedding(num_nodes, num_feats)  # torch.LongTensor等价于torch.tensor
    features.weight = nn.Parameter(torch.FloatTensor(feat_data),
                                   requires_grad=False)  # torch.FloatTensor等价于torch.Tensor
    # 上一行的作用：加快训练速度，提升训练效果(F1)
    # print(features)
    # print(features.weight.sum(dim=1))

    agg1 = MeanAggregator(features)
    enc1 = Encoder(features=features, feature_dim=1433, embed_dim=128, adj_lists=adj_lists,
                   aggregator=agg1)  # gcn=False
    agg2 = MeanAggregator(features=lambda nodes: enc1(nodes).t())
    enc2 = Encoder(features=lambda nodes: enc1(nodes).t(), feature_dim=enc1.embed_dim, embed_dim=128,
                   adj_lists=adj_lists, aggregator=agg2, base_model=enc1)  # gcn=False

    graphsage = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    # print(num_nodes)
    # print(rand_indices)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
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
        # print((val_output.max(1)[1].reshape(labels[val].shape) == labels[val]).sum())
        acc_val = accuracy(val_output, labels[val])
        print("epoch:{}".format(batch + 1), "loss:{}".format(loss.item()), "accuracy:{}".format(acc_val))

    # print(val_output.data.numpy().argmax(axis=1))
    # print("-"*50)
    # print(val_output.argmax(axis=1))
    # print("Validation F1:", f1_score(labels[val], val_output.argmax(axis=1), average="micro"))
    # print("Average batch time:", np.mean(times))

if __name__ =="__main__":
    # graph_train()
    cora_train()