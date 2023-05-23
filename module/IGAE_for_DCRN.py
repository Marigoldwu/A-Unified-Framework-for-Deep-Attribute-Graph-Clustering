# -*- coding: utf-8 -*-
"""
@Time: 2023/5/23 18:23 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# GNNLayer from DFCN
class GNNLayer(nn.Module):
    def __init__(self, name, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        if name == "dblp":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        else:
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            if self.name == "dblp":
                support = self.act(F.linear(features, self.weight))
            else:
                support = self.act(torch.mm(features, self.weight))
        else:
            if self.name == "dblp":
                support = F.linear(features, self.weight)
            else:
                support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        az = torch.spmm(adj, output)
        return output, az


# IGAE encoder from DFCN
class IGAE_encoder(nn.Module):
    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(name, n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(name, gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(name, gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1, az_1 = self.gnn_1(x, adj, active=True)
        z_2, az_2 = self.gnn_2(z_1, adj, active=True)
        z_igae, az_3 = self.gnn_3(z_2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj, [az_1, az_2, az_3], [z_1, z_2, z_igae]


# IGAE decoder from DFCN
class IGAE_decoder(nn.Module):
    def __init__(self, name, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(name, gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(name, gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(name, gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_1, az_1 = self.gnn_4(z_igae, adj, active=True)
        z_2, az_2 = self.gnn_5(z_1, adj, active=True)
        z_hat, az_3 = self.gnn_6(z_2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2, az_3], [z_1, z_2, z_hat]


# Improved Graph Auto Encoder from DFCN
class IGAE(nn.Module):
    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        # IGAE encoder
        self.encoder = IGAE_encoder(name,
                                    gae_n_enc_1=gae_n_enc_1,
                                    gae_n_enc_2=gae_n_enc_2,
                                    gae_n_enc_3=gae_n_enc_3,
                                    n_input=n_input)

        # IGAE decoder
        self.decoder = IGAE_decoder(name,
                                    gae_n_dec_1=gae_n_dec_1,
                                    gae_n_dec_2=gae_n_dec_2,
                                    gae_n_dec_3=gae_n_dec_3,
                                    n_input=n_input)
