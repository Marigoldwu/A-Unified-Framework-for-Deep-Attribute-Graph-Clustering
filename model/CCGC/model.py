# -*- coding: utf-8 -*-
"""
@Time: 2023/6/28 14:25 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch.nn as nn
import torch.nn.functional as F


class Encoder_Net(nn.Module):
    def __init__(self, dims):
        super(Encoder_Net, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])

    def forward(self, x):
        out1 = self.layers1(x)
        out2 = self.layers2(x)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2
