# -*- coding: utf-8 -*-
"""
@Time: 2023/5/8 17:25 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.GCN import GCN


class GAE(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_dim):
        super(GAE, self).__init__()
        self.gcn1 = GCN(input_dim, hidden_size)
        self.gcn2 = GCN(hidden_size, embedding_dim, activeType='no')

    def forward(self, x, adj):
        h1 = self.gcn1(x, adj)
        h2 = self.gcn2(h1, adj)
        embedding = F.normalize(h2, p=2, dim=1)
        A_pred = dot_product_decode(embedding)
        return A_pred, embedding


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred
