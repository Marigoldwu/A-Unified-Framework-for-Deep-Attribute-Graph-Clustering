# -*- coding: utf-8 -*-
"""
@Time: 2023/4/29 16:54 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn.parameter import Parameter
from module.AE import AE
from module.GCN import GCN


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)

        return weight_output


class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)

        return weight_output


class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)

        return weight_output


class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)

        return weight_output


class AGCN(nn.Module):

    def __init__(self, input_dim, embedding_dim, enc_1_dim, enc_2_dim, enc_3_dim, dec_1_dim, dec_2_dim, dec_3_dim,
                 clusters, v=1):
        super(AGCN, self).__init__()

        # AE
        self.ae = AE(input_dim,
                     embedding_dim,
                     enc_1_dim,
                     enc_2_dim,
                     enc_3_dim,
                     dec_1_dim,
                     dec_2_dim,
                     dec_3_dim)

        self.agcn_0 = GCN(input_dim, enc_1_dim, activeType='leaky_relu')
        self.agcn_1 = GCN(enc_1_dim, enc_2_dim, activeType='leaky_relu')
        self.agcn_2 = GCN(enc_2_dim, enc_3_dim, activeType='leaky_relu')
        self.agcn_3 = GCN(enc_3_dim, embedding_dim, activeType='leaky_relu')
        self.agcn_z = GCN(3020, clusters, activeType="no")

        self.mlp = MLP_L(3020)

        # attention on [Z_i || H_i]
        self.mlp1 = MLP_1(2 * enc_1_dim)
        self.mlp2 = MLP_2(2 * enc_2_dim)
        self.mlp3 = MLP_3(2 * enc_3_dim)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # AE Module
        x_bar, h1, h2, h3, z = self.ae(x)

        x_array = list(np.shape(x))
        n_x = x_array[0]

        # AGCN-H
        z1 = self.agcn_0(x, adj)
        # z2
        m1 = self.mlp1(torch.cat((h1, z1), 1))
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:, 0], [n_x, 1])
        m12 = torch.reshape(m1[:, 1], [n_x, 1])
        m11_broadcast = m11.repeat(1, 500)
        m12_broadcast = m12.repeat(1, 500)
        z2 = self.agcn_1(m11_broadcast.mul(z1) + m12_broadcast.mul(h1), adj)
        # z3
        m2 = self.mlp2(torch.cat((h2, z2), 1))
        m2 = F.normalize(m2, p=2)
        m21 = torch.reshape(m2[:, 0], [n_x, 1])
        m22 = torch.reshape(m2[:, 1], [n_x, 1])
        m21_broadcast = m21.repeat(1, 500)
        m22_broadcast = m22.repeat(1, 500)
        z3 = self.agcn_2(m21_broadcast.mul(z2) + m22_broadcast.mul(h2), adj)
        # z4
        m3 = self.mlp3(torch.cat((h3, z3), 1))  # self.mlp3(h2)
        m3 = F.normalize(m3, p=2)
        m31 = torch.reshape(m3[:, 0], [n_x, 1])
        m32 = torch.reshape(m3[:, 1], [n_x, 1])
        m31_broadcast = m31.repeat(1, 2000)
        m32_broadcast = m32.repeat(1, 2000)
        z4 = self.agcn_3(m31_broadcast.mul(z3) + m32_broadcast.mul(h3), adj)

        # # AGCN-S
        u = self.mlp(torch.cat((z1, z2, z3, z4, z), 1))
        u = F.normalize(u, p=2)
        u0 = torch.reshape(u[:, 0], [n_x, 1])
        u1 = torch.reshape(u[:, 1], [n_x, 1])
        u2 = torch.reshape(u[:, 2], [n_x, 1])
        u3 = torch.reshape(u[:, 3], [n_x, 1])
        u4 = torch.reshape(u[:, 4], [n_x, 1])

        tile_u0 = u0.repeat(1, 500)
        tile_u1 = u1.repeat(1, 500)
        tile_u2 = u2.repeat(1, 2000)
        tile_u3 = u3.repeat(1, 10)
        tile_u4 = u4.repeat(1, 10)

        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1)
        embedding = self.agcn_z(net_output, adj)
        predict = F.softmax(embedding, dim=1)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, embedding
