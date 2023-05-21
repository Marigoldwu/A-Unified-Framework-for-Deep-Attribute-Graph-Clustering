# -*- coding: utf-8 -*-
"""
@Time: 2023/5/20 16:14 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from module.AE_for_DFCN import AE
from module.IGAE import IGAE


class AE_IGAE(nn.Module):
    def __init__(self, name, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_node=None):
        super(AE_IGAE, self).__init__()
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
        return x_hat, adj_hat, z_hat, z_tilde
