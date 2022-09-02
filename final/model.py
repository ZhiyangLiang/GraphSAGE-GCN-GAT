import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter # 也可直接调用nn.Parameter
from torch.nn.modules.module import Module # 也可直接调用nn.Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # 注意：torch.FloatTensor生成的元素数值非常接近0;torch.Long生成的元素数值非常大
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight2 = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()  # 此处表示生成变量后(即上面的语句运行后),将会进行变量初始化(即执行该语句)

    def reset_parameters(self):  # 经测试,重写可覆盖
        stdv = 1/math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight2.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)    # 尝试：此行与下一行交换顺序
        # output = torch.spmm(adj, support) # torch.spmm支持sparse在前,dense在后的矩阵乘法
        # support = torch.spmm(adj, input)
        support = torch.mm(adj, input)    # 返回的adj为dense
        output = torch.mm(support, self.weight)
        if self.bias is not None:         # adj是稀疏矩阵
            return output +self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__+"("+str(self.in_features)+"->"+str(self.out_features)+")"


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # 尝试："1"应该换为一个超参数t
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape:(N, in_features), Wh.shape:(N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)  # 为什么要用广播机制?
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # a[:self.out_features, :]用于被处理的节点i
        # a[self.out_features:, :]用于节点i邻域内的节点j
        e = Wh1 * Wh2.T  # broadcast add 注：源代码用的是"+" 尝试：对比二者效果
        # e为N个节点中任意两个节点之间的相关度组成的矩阵(N*N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.in_features) + "->" + str(self.out_features) + ")"


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 每个attention的输出维度为8, 8个attention拼接即得64维
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # out_att的输入维度为64, 输出维度为7, 即种类数
        return F.log_softmax(x, dim=1)