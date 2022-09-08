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

class MeanAggregator(nn.Module):
    def __init__(self, features, cuda=False, gcn=False):
        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=None):
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
            # 注意：此处先执行前面的判断语句，后执行for to_neigh in to_neighs形成list
        else:
            samp_neighs = to_neighs
        if self.gcn:
            samp_neighs = [set.union(samp_neigh, _set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdims=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats


class Encoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=10, base_model=None, gcn=False, cuda=False):
        super(Encoder, self).__init__()
        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2*self.feat_dim))
        # 注意：此处将","后面的语句作为整体执行
        init.xavier_uniform_(self.weight) # 初始化操作

    def forward(self, nodes):
        # neigh_feats = self.aggregator.forward(nodes, self.adj_lists, self.num_sample)
        # print(self.adj_lists)
        # print("-"*50)
        # print([self.adj_lists[int(node)] for node in nodes])
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
                # print(self_feats.shape) # (34, 16)
            combined = torch.cat([self_feats, neigh_feats], dim=1)
            # print(combined.shape) # (34, 32)
        else:
            combined = neigh_feats
        # print(combined.shape) # (34, 32)
        # print(self.weight.shape) # (16, 32)
        combined = F.relu(self.weight.mm(combined.t()))
        # print(combined.shape) # (16, 34)
        return combined


class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        # print(embeds.shape) # (16, 34)
        # print(self.weight.shape) # (4, 16)
        scores = self.weight.mm(embeds)
        # print(scores.shape) # (4, 34)
        # return scores.t() # (34, 4)
        scores_softmax = torch.exp(scores.t())/torch.sum(torch.exp(scores.t()), dim=1).reshape(-1, 1)
        # print(scores_softmax.shape)
        # print(scores_softmax.sum(dim=1))
        return scores_softmax  # (34, 4)

    def loss(self, nodes, labels):
        print("nodes:", nodes)
        scores = self.forward(nodes)
        # print(scores.sum(dim=1))
        print("scores.shape:", scores.shape)
        print("labels.shape:", labels.shape)
        print("scores:", scores)
        print("labels:", labels)
        return self.xent(scores, labels)


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
        nodes = []
        for i in range(len(graph.nodes)):
            nodes.append(i)
    split = int(len(nodes)/3*2)
    features = nn.Embedding(num_embed, embed_dim)
    to_neighs = get_to_neighs(graph)
    aggregator = MeanAggregator(features, nodes[:split], to_neighs)
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