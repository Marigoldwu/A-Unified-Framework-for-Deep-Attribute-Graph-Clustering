# -*- coding: utf-8 -*-
"""
@Time: 2023/5/20 10:43 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


class GNNLayer(Module):

    def __init__(self, name, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        if name == "dblp" or name == "hhar":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        elif name == "usps" or name == "acm" or name == "cite" or name == "cora":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        elif name == "reut":
            self.act = nn.LeakyReLU(0.2, inplace=True)
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            if self.name == "dblp" or self.name == "hhar":
                support = self.act(F.linear(features, self.weight))
            else:
                support = self.act(torch.mm(features, self.weight))
        else:
            if self.name == "dblp" or self.name == "hhar":
                support = F.linear(features, self.weight)
            else:
                support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        return output


class IGAE_encoder(nn.Module):

    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.name = name
        self.gnn_1 = GNNLayer(name, n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(name, gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(name, gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=False if self.name == "hhar" else True)
        z = self.gnn_2(z, adj, active=False if self.name == "hhar" else True)
        z_igae = self.gnn_3(z, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):

    def __init__(self, name, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.name = name
        self.gnn_4 = GNNLayer(name, gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(name, gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(name, gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj, active=False if self.name == "hhar" else True)
        z = self.gnn_5(z, adj, active=False if self.name == "hhar" else True)
        z_hat = self.gnn_6(z, adj, active=False if self.name == "hhar" else True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class IGAE(nn.Module):

    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(name=name,
                                    gae_n_enc_1=gae_n_enc_1,
                                    gae_n_enc_2=gae_n_enc_2,
                                    gae_n_enc_3=gae_n_enc_3,
                                    n_input=n_input)

        self.decoder = IGAE_decoder(name=name,
                                    gae_n_dec_1=gae_n_dec_1,
                                    gae_n_dec_2=gae_n_dec_2,
                                    gae_n_dec_3=gae_n_dec_3,
                                    n_input=n_input)

    def forward(self, x, adj):
        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat
