# -*- coding: utf-8 -*-
"""
import torch.nn as nn
from module.AE import AE
from module.GAE import GAE


class Example(nn.Module):
    def __init__(self):
        super(Example, self).__init__()
        self.ae = AE(args)  # args are parameters of AE
        self.gae = GAE(args)  # args are parameters of GAE

    def forward(self, x, adj):
        x_bar, enc1, enc2, enc3, embedding_ae = self.ae(x)
        a_bar, embedding_gae = self.gae(x, adj)

        q = ...
    return x_bar, ..., q

"""