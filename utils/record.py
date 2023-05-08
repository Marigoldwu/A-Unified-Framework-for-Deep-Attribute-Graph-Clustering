# -*- coding: utf-8 -*-
"""
@Time: 2023/5/8 10:54 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""


def record_metrics(acc_list, nmi_list, ari_list, f1_list, metrics):
    acc_list.apeend(metrics[0])
    nmi_list.apeend(metrics[1])
    ari_list.apeend(metrics[2])
    f1_list.apeend(metrics[3])
    return acc_list, nmi_list, ari_list, f1_list
