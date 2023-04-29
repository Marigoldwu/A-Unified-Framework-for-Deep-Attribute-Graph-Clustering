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

from module.GAT import GraphAttentionLayer


class GAE(nn.Module):
    def __init__(self, n_input):
        super(GAE, self).__init__()
        self.conv1 = GraphAttentionLayer(n_input, 500)
        self.conv2 = GraphAttentionLayer(500, 500)
        self.conv3 = GraphAttentionLayer(500, 2000)
        self.conv4 = GraphAttentionLayer(2000, 10)

    def forward(self, x, adj, M):
        r1 = self.conv1(x, adj, M)
        r2 = self.conv2(r1, adj, M)
        r3 = self.conv3(r2, adj, M)
        r4 = self.conv4(r3, adj, M)
        r = F.normalize(r4, p=2, dim=1)
        A_pred = torch.sigmoid(torch.matmul(r, r.t()))
        return A_pred, r
