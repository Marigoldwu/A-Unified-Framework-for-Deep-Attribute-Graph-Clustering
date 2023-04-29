# -*- coding: utf-8 -*-
"""
@Time: 2023/4/28 14:16 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from module.GATLayer import GraphAttentionLayer


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred, z
