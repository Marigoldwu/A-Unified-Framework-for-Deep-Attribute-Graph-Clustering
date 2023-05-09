# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

import os
import sys
import torch
import logging
import numpy as np
from torch.utils.data import Dataset

from utils.data_processor import numpy_to_torch, construct_graph, normalize_adj, get_M


class Data:
    def __init__(self, feature, label, adj, M):
        self.feature = feature
        self.label = label
        self.adj = adj
        self.M = M

    def __getattr__(self, name):
        if name == 'feature':
            return self.feature
        elif name == 'label':
            return self.label
        elif name == 'adj':
            return self.adj
        elif name == 'M':
            return self.M
        else:
            raise AttributeError(f"'Data' object has no attribute '{name}'")


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def load_graph_data(root_path=".", dataset_name="dblp", show_details=False):
    """
    load graph data
    :param root_path: the root path
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :returns feat, label, adj: the features, labels and adj
    """
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        stream=sys.stdout)
    root_path = root_path + "dataset/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dataset_path = root_path + dataset_name
    if not os.path.exists(dataset_path):
        # down load
        url = "https://drive.google.com/file/d/1_LesghFTQ02vKOBUfDP8fmDF1JP3MPrJ/view?usp=sharing"
        logging.info("Downloading " + dataset_name + " dataset from: " + url)
    else:
        logging.info("Loading " + dataset_name + " dataset from local")
    load_path = root_path + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)

    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("edge num:   ", int(adj.sum() / 2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj


def load_non_graph_data(root_path="./", dataset_name="USPS", show_details=False):
    """
    load non-graph data
    :param root_path: the root path
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - category num
    - category distribution
    :returns feat, label: the features and labels
    """
    root_path = root_path + "dataset/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dataset_path = root_path + dataset_name
    if not os.path.exists(dataset_path):
        # down load
        pass
    load_path = root_path + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)

    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("------details of dataset------")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("category num:   ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label


def load_data(k, dataset_path, dataset_name,
              feature_type="tensor", adj_type="tensor", label_type="npy",
              adj_loop=True, adj_norm=False, adj_symmetric=True, t=None):
    """
    load feature, label, adj, M according to the value of k.
    If k is None, then load graph data, otherwise load non-graph data.
    If cal_M is False, M is still set to None to remain the consistency of the number of function return values.
    Meanwhile, you can specify the datatype as 'tensor' or 'npy'.

    :param k: To distinguish the data is graph data or non-graph data. 'None' denotes graph and int denotes non-graph.
    :param dataset_path: The store path of dataset.
    :param dataset_name: The name of dataset.
    :param feature_type: The datatype of feature. 'tensor' and 'npy' are available.
    :param adj_type: The datatype of adj. 'tensor' and 'npy' are available.
    :param label_type: The datatype of label. 'tensor' and 'npy' are available.
    :param adj_loop: Whether the adj has self-loop. If the value is True, the elements at the diagonal position is 1.
    :param adj_norm: Whether to normalize the adj. Default is False.
    :param adj_symmetric: Whether the normalization type is symmetric.
    :param t: t in the formula of M
    :return: feature, label, adj, M

    """
    # If the dataset is not graph dataset, use construct_graph to get KNN graph.
    if k is None:
        feature, label, adj = load_graph_data(dataset_path, dataset_name)
    else:
        feature, label = load_non_graph_data(dataset_path, dataset_name)
        metric_dict = {"usps": "heat", "hhar": "cosine", "reut": "cosine"}
        adj = construct_graph(feature, k, metric_dict[dataset_name])
    # Whether the adj has self-loop, default is True.
    if adj_loop:
        adj = adj + np.eye(adj.shape[0])
    # Whether calculate the matrix M
    M = None
    if t is not None:
        M = get_M(adj, t)
    # normalize the adj
    if adj_norm:
        adj = normalize_adj(adj, adj_symmetric)
    # transform the datatype
    if feature_type == "tensor":
        feature = numpy_to_torch(feature)
    if adj_type == "tensor":
        adj = numpy_to_torch(adj)
    if label_type == "tensor":
        label = numpy_to_torch(label)
    data = Data(feature, label, adj, M)
    return data
