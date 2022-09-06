import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

def load_data2():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))  # 此处不能用torch代替np
    labels = np.empty((num_nodes, 1), dtype=np.int64)  # 此处不能用torch代替np
    node_map = {}
    label_map = {}
    with open("./cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            info_label = info[-1]
            info = info[:-1]
            info = [int(x) for x in info]
            feat_data[i, :] = info[1:]
            # print(len(feat_data[i, :]))
            node_map[info[0]] = i
            if not info_label in label_map:
                label_map[info_label] = len(label_map)  # len({}) = 0
            labels[i] = label_map[info_label]

    adj_lists = defaultdict(set)  # 注意：此处adj_lists不能用普通的set
    with open("./cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            info = [int(x) for x in info]
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c:np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="./cora/", dataset="cora"):
    print("Loading {} dataset...".format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    labels2 = idx_features_labels[:, -1]

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j:i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T>adj)  # 最后减的那一项目的是去除负边
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    # idx_train = range(1000)
    # idx_val = range(1000, 1500)
    # idx_test = range(1500, 2700)

    # adj = sparse_mx_to_torch_sparse_tensor(adj) # 区别于GAT
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)  # 此处得到一个对角矩阵
    mx = r_mat_inv.dot(mx)  # 注意.dot为矩阵乘法,不是对应元素相乘
    return mx

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    # return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    mid = np.dot(r_mat_inv_sqrt, mx)
    return np.dot(mid, r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse.FloatTensor(indices, values, shape)