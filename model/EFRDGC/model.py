# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 9:35 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from module.AE import AE
from module.GAE_for_EFRDGC import GAE


class EFRDGC(nn.Module):
    def __init__(self, input_dim, embedding_dim, enc_1_dim, enc_2_dim, enc_3_dim,
                 dec_1_dim, dec_2_dim, dec_3_dim, hidden_1_dim, hidden_2_dim, hidden_3_dim, clusters, alpha=0.2, v=1):
        super(EFRDGC, self).__init__()
        self.ae = AE(input_dim, embedding_dim,
                     enc_1_dim, enc_2_dim, enc_3_dim,
                     dec_1_dim, dec_2_dim, dec_3_dim)

        self.gae = GAE(input_dim, hidden_1_dim, hidden_2_dim, hidden_3_dim, alpha)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(clusters, hidden_3_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

    def forward(self, x, adj, M, sigma=0.1):
        x_bar, enc_h1, enc_h2, enc_h3, z = self.ae(x)
        A_pred, z = self.gae(x, adj, M, enc_h1, enc_h2, enc_h3, sigma)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return A_pred, z, q, x_bar
