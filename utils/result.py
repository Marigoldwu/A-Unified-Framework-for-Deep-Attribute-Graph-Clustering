# -*- coding: utf-8 -*-
"""
@Time: 2023/5/25 8:36 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""


class Result:
    def __init__(self, embedding=None, max_acc_corresponding_metrics=None, loss=None,
                 acc=None, acc_p=None, acc_q=None, acc_q_ae=None, acc_q_igae=None):
        self.embedding = embedding
        self.max_acc_corresponding_metrics = max_acc_corresponding_metrics
        self.loss = loss
        self.acc = acc
        self.acc_p = acc_p
        self.acc_q = acc_q
        self.acc_q_ae = acc_q_ae
        self.acc_q_igae = acc_q_igae
