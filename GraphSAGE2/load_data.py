import numpy as np
from collections import defaultdict

def load_cora():
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