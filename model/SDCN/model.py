# -*- coding: utf-8 -*-
"""
@Time: 2023/4/28 16:28 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn

from module.AE import AE
from module.GCN import GCN
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SDCN(nn.Module):

    def __init__(self, input_dim, embedding_dim, enc_1_dim, enc_2_dim, enc_3_dim,
                 dec_1_dim, dec_2_dim, dec_3_dim, clusters, v=1):
        super(SDCN, self).__init__()

        # auto-encoder for intra information
        self.ae = AE(input_dim=input_dim,
                     embedding_dim=embedding_dim,
                     enc_1_dim=enc_1_dim,
                     enc_2_dim=enc_2_dim,
                     enc_3_dim=enc_3_dim,
                     dec_1_dim=dec_1_dim,
                     dec_2_dim=dec_2_dim,
                     dec_3_dim=dec_3_dim)

        # GCN for inter information
        self.gnn_1 = GCN(input_dim, enc_1_dim)
        self.gnn_2 = GCN(enc_1_dim, enc_2_dim)
        self.gnn_3 = GCN(enc_2_dim, enc_3_dim)
        self.gnn_4 = GCN(enc_3_dim, embedding_dim)
        self.gnn_5 = GCN(embedding_dim, clusters, activeType="no")

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5
        # GCN Module
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2(((1 - sigma) * h1 + sigma * tra1), adj)
        h3 = self.gnn_3(((1 - sigma) * h2 + sigma * tra2), adj)
        h4 = self.gnn_4(((1 - sigma) * h3 + sigma * tra3), adj)
        h5 = self.gnn_5(((1 - sigma) * h4 + sigma * z), adj)
        predict = F.softmax(h5, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z
