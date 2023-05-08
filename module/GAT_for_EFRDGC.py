# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 9:47 
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
    def __init__(self, n_input, hidden_1_dim, hidden_2_dim, hidden_3_dim, alpha):
        super(GAT, self).__init__()
        self.conv1 = GraphAttentionLayer(n_input, hidden_1_dim, alpha)
        self.conv2 = GraphAttentionLayer(hidden_1_dim, hidden_2_dim, alpha)
        self.conv3 = GraphAttentionLayer(hidden_2_dim, hidden_3_dim, alpha)

    def forward(self, x, adj, M, enc_h1, enc_h2, enc_h3, sigma=0.5):
        r1 = self.conv1(x, adj, M)
        r2 = self.conv2((1-sigma) * r1 + sigma * enc_h1, adj, M)
        r3 = self.conv3((1-sigma) * r2 + sigma * enc_h2, adj, M)
        r = F.normalize((1-sigma) * r3 + sigma * enc_h3, p=2, dim=1)
        A_pred = torch.sigmoid(torch.matmul(r, r.t()))
        return A_pred, r
