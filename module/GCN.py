# -*- coding: utf-8 -*-
"""
@Time: 2022/3/19 15:25
@Author: Marigold
@Version: 1.0.0
@Description：图卷积神经网络模块
@WeChat Account: Marigold
"""

import torch
from torch.nn import ReLU, LeakyReLU
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCN(Module):
    def __init__(self, in_features, out_features, activeType='relu'):
        """
        :param in_features: the dimension of input features
        :param out_features: the dimension of output features
        :param activeType: The activation function is used by default,
                            and the default type of activation function is 'relu',
                            or you can set activateType 'leaky_relu'.
                            If you don't want to use activation function, please set the activateType 'no'.
        """
        super(GCN, self).__init__()
        self.isActivate = True
        if activeType == 'leaky_relu':
            self.activate = LeakyReLU(negative_slope=0.2)
        elif activeType == 'relu':
            self.activate = ReLU()
        else:
            self.isActivate = False
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj):
        """
        :param features: the input features
        :param adj: the Symmetric normalized adjacent matrix
        """
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if self.isActivate:
            activated_output = self.activate(output)
            return activated_output
        return output
