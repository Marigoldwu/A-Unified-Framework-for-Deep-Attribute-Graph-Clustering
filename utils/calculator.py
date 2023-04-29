# -*- coding: utf-8 -*-
"""
@Time: 2023/4/28 16:13 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import numpy as np


def cal_mean_std(acc, nmi, ari, f1):
    acc_mean = np.mean(acc)
    nmi_mean = np.mean(nmi)
    ari_mean = np.mean(ari)
    f1_mean = np.mean(f1)
    acc_std = np.std(acc)
    nmi_std = np.std(nmi)
    ari_std = np.std(ari)
    f1_std = np.std(f1)
    result = "acc: {:0>.4f}±{:0>.4f}\t\tnmi: {:0>.4f}±{:0>.4f}\t\tari: {:0>.4f}±{:0>.4f}\t\tf1: {:0>.4f}±{:0>.4f}"\
        .format(acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std)
    return result
