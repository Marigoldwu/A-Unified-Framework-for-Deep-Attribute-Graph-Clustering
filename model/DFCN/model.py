# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 14:53 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from module.AE_for_DFCN import AE
from module.IGAE import IGAE


class DFCN(nn.Module):

    def __init__(self, name, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, v=1.0, n_node=None):
        super(DFCN, self).__init__()

        self.ae = AE(ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z)

        self.igae = IGAE(name=name,
                         gae_n_enc_1=gae_n_enc_1,
                         gae_n_enc_2=gae_n_enc_2,
                         gae_n_enc_3=gae_n_enc_3,
                         gae_n_dec_1=gae_n_dec_1,
                         gae_n_dec_2=gae_n_dec_2,
                         gae_n_dec_3=gae_n_dec_3,
                         n_input=n_input)

        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).cuda()
        self.b = 1 - self.a

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.igae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.mm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.igae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_tilde.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde
