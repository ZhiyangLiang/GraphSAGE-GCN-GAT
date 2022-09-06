import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    def __init__(self, cuda=False, gcn=False):
        super(MeanAggregator, self).__init__()
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, features, adj):
        # _set = set
        # if not num_sample is None:
        #     _sample = random.sample
        #     samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        #     # 注意：此处先执行前面的判断语句，后执行for to_neigh in to_neighs形成list
        # else:
        #     samp_neighs = to_neighs
        # if self.gcn:
        #     samp_neighs = [set.union(samp_neigh, _set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
        # unique_nodes_list = list(set.union(*samp_neighs))
        # unique_nodes = {n:i for i, n in enumerate(unique_nodes_list)}
        # mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # mask[row_indices, column_indices] = 1
        # if self.cuda:
        #     mask = mask.cuda()
        # num_neigh = mask.sum(1, keepdims=True)
        # mask = mask.div(num_neigh)
        # if self.cuda:
        #     embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        # else:
        #     embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = adj.mm(features)
        return to_feats


class Encoder(nn.Module):
    def __init__(self, feature_dim, embed_dim, aggregator, base_model=None, gcn=False, cuda=False):
        super(Encoder, self).__init__()
        self.feat_dim = feature_dim
        self.aggregator = aggregator
        self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim if self.gcn else 2*self.feat_dim, embed_dim))
        init.xavier_uniform_(self.weight)  # 初始化操作

    def forward(self, features, adj):
        if self.base_model != None:
            features = self.base_model(features, adj)
        neigh_feats = self.aggregator.forward(features, adj)
        if not self.gcn:
            if self.cuda:
                self_feats = features.cuda()
            else:
                self_feats = features
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(combined.mm(self.weight))
        return combined


class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, dropout, enc):
        super(SupervisedGraphSage, self).__init__()
        self.dropout = dropout
        self.enc = enc
        self.weight = nn.Parameter(torch.FloatTensor(enc.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)

    def forward(self, features, adj):
        embeds = self.enc(features, adj)
        # embeds = F.relu(self.enc(features, adj))
        scores = embeds.mm(self.weight)
        # scores = F.dropout(scores, self.dropout, training=self.training)
        return F.log_softmax(scores, dim=1)