# -*- coding: utf-8 -*-
"""
@Time: 2022/12/2 12:54 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch.nn.functional as F
from torch.nn import Linear, Module


class AE(Module):
    def __init__(self, input_dim,
                 embedding_dim,
                 enc_1_dim,
                 enc_2_dim,
                 enc_3_dim,
                 dec_1_dim,
                 dec_2_dim,
                 dec_3_dim):
        """
        :param input_dim: the dimension of input data
        :param embedding_dim: the dimension of embedding features
        :param enc_1_dim: the dimension of the 1st layer of encoder
        :param enc_2_dim: the dimension of the 2nd layer of encoder
        :param enc_3_dim: the dimension of the 3rd layer of encoder
        :param dec_1_dim: the dimension of the 1st layer of decoder
        :param dec_2_dim: the dimension of the 2nd layer of decoder
        :param dec_3_dim: the dimension of the 3rd layer of decoder
        """
        super(AE, self).__init__()
        self.enc_1 = Linear(input_dim, enc_1_dim)
        self.enc_2 = Linear(enc_1_dim, enc_2_dim)
        self.enc_3 = Linear(enc_2_dim, enc_3_dim)
        self.z_layer = Linear(enc_3_dim, embedding_dim)

        self.dec_1 = Linear(embedding_dim, dec_1_dim)
        self.dec_2 = Linear(dec_1_dim, dec_2_dim)
        self.dec_3 = Linear(dec_2_dim, dec_3_dim)
        self.x_bar_layer = Linear(dec_3_dim, input_dim)

    def forward(self, x):
        """

        :param x:
        :return:
        - x_bar: the reconstructed features
        - enc_h1: the 1st layers features of encoder
        - enc_h2: the 2nd layers features of encoder
        - enc_h3: the 3rd layers features of encoder
        - z: the embedding
        """
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z
