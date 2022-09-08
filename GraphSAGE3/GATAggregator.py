import time
import random
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