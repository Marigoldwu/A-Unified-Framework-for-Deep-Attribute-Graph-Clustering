# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 18:50 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""


def count_parameters(model):
    """
    count the parameters' number of the input model
    Note: The unit of return value is millions(M) if exceeds 1,000,000.
    :param model:
    :return:
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(num_params / 1e6, 6)
