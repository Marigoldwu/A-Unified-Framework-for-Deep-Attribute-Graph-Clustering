# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 17:12 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from module.GAE import GAE


class GCAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, embedding_dim, v=1.0):
        super(GCAE, self).__init__()
        self.gae = GAE(input_dim, hidden_size, embedding_dim)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(output_dim, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

    def forward(self, x, adj):
        A_pred, embedding = self.gae(x, adj)

        q = 1.0 / (1.0 + torch.sum(torch.pow(embedding.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return A_pred, q, embedding


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred
