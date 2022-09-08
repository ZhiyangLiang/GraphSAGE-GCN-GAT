import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

class GATAggregator(nn.Module):
    def __init__(self, features, features_dim, cuda=False, gcn=False):
        super(GATAggregator, self).__init__()
        self.features = features
        self.features_dim = features_dim
        self.cuda = cuda
        self.gcn = gcn
        self.a = nn.Parameter(torch.empty(size=(2 * features_dim, 1)))

    def forward(self, nodes, to_neighs, num_sample=10):
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
            # 注意：此处先执行前面的判断语句，后执行for to_neigh in to_neighs形成list
        else:
            samp_neighs = to_neighs
        # if self.gcn:
        #     samp_neighs = [set.union(samp_neigh, _set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
        samp_neighs = [set.union(samp_neigh, _set([nodes[i].item()])) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        # print("unique_nodes_list:", unique_nodes_list)
        unique_nodes = {n:i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdims=True)
        # mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        # to_feats = mask.mm(embed_matrix)
        Wh1 = torch.matmul(embed_matrix, self.a[:self.features_dim, :])
        Wh2 = torch.matmul(embed_matrix, self.a[self.features_dim:, :])
        # print("a1.shape:", self.a[:self.features_dim, :].shape)
        # print("a2.shape:", self.a[self.features_dim:, :].shape)
        # print("Wh1.shape:", Wh1)
        # print("Wh2.shape:", Wh2)
        e = Wh1 + Wh2.T
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, 0.5, training=self.training)
        # print("attention.shape", attention.shape)
        mid1 = torch.tensor([unique_nodes[i.item()] for i in nodes])
        # print("mid1:", mid1)
        mid2 = attention[mid1]
        # print("mid2:", mid2)
        # print("mid2.shape:", mid2.shape)
        # print("mask.shape:", mask.shape)
        mid3 = mid2 * mask
        return torch.mm(mid3, embed_matrix)

class GCNAggregator(nn.Module):
    def __init__(self, features, cuda=False, gcn=False):
        super(GCNAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
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
        # to_feats = mask.mm(embed_matrix)
        rowsum = mask.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        colsum = mask.sum(0)
        c_inv_sqrt = torch.pow(colsum, -0.5).flatten()
        c_inv_sqrt[torch.isinf(c_inv_sqrt)] = 0.
        c_mat_inv_sqrt = torch.diag(c_inv_sqrt)
        # print("mask.shape:", mask.shape)
        # print("r_mat_inv_sqrt.shape:", r_mat_inv_sqrt.shape)
        # print("c_mat_inv_sqrt.shape:", c_mat_inv_sqrt.shape)
        # print("embed_matrix.shape:", embed_matrix.shape)
        mid1 = torch.mm(r_mat_inv_sqrt, mask)
        mid2 = torch.mm(mid1, c_mat_inv_sqrt)
        return torch.mm(mid2, embed_matrix)

class MaxAggregator(nn.Module):
    def __init__(self, features, cuda=False, gcn=False):
        super(MaxAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
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
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        # to_feats = mask.mm(embed_matrix)
        to_feats = torch.zeros(mask.shape[0], embed_matrix.shape[1])
        for i in range(mask.shape[0]):
            mid = mask[i] * embed_matrix.T
            to_feats[i] = mid.max(1)[0]
        return to_feats

class SumAggregator(nn.Module):
    def __init__(self, features, cuda=False, gcn=False):
        super(SumAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
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
        # mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class MeanAggregator(nn.Module):
    def __init__(self, features, cuda=False, gcn=False):
        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
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
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=10, base_model=None, gcn=False, cuda=False, feature_transform=False):
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
        init.xavier_uniform_(self.weight)  # 初始化操作

    def forward(self, nodes):
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
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
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)

        # preds = scores.max(1)[1].type_as(labels).reshape(labels.shape)
        # correct = preds.eq(labels).double()
        # correct = correct.sum()
        # return self.xent(scores, labels), correct / len(labels)

        return self.xent(scores, labels)

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