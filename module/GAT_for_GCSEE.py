# -*- coding: utf-8 -*-
"""
@Time: 2022/12/5 16:50 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
from torch import nn

from module.GATLayer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, n_input, embedding_size, hidden_1_dim, hidden_2_dim, hidden_3_dim, alpha):
        super(GAT, self).__init__()
        self.conv1 = GraphAttentionLayer(n_input, hidden_1_dim, alpha)
        self.conv2 = GraphAttentionLayer(hidden_1_dim, hidden_2_dim, alpha)
        self.conv3 = GraphAttentionLayer(hidden_2_dim, hidden_3_dim, alpha)
        self.conv4 = GraphAttentionLayer(hidden_3_dim, embedding_size, alpha)

    def forward(self, x, adj, M):
        r1 = self.conv1(x, adj, M)
        r2 = self.conv2(r1, adj, M)
        r3 = self.conv3(r2, adj, M)
        r4 = self.conv4(r3, adj, M)
        r = F.normalize(r4, p=2, dim=1)
        A_pred = torch.sigmoid(torch.matmul(r, r.t()))
        return A_pred, r
