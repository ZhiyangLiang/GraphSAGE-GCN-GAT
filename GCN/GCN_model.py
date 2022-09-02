import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter  # 也可直接调用nn.Parameter
from torch.nn.modules.module import Module  # 也可直接调用nn.Module


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