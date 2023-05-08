# -*- coding: utf-8 -*-
"""
@Time: 2023/5/8 14:04 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import numpy as np


def cal_mean_std(acc_list, nmi_list, ari_list, f1_list):
    """
    calculate the mean and standard deviation
    :param acc_list: the list including acc from each iteration
    :param nmi_list: the list including acc from each iteration
    :param ari_list: the list including acc from each iteration
    :param f1_list: the list including acc from each iteration
    :return: a formatted string including acc, nmi, ari, f1 with the format: mean±std
    """
    acc_mean = np.mean(acc_list)
    nmi_mean = np.mean(nmi_list)
    ari_mean = np.mean(ari_list)
    f1_mean = np.mean(f1_list)
    acc_std = np.std(acc_list)
    nmi_std = np.std(nmi_list)
    ari_std = np.std(ari_list)
    f1_std = np.std(f1_list)
    result = f"acc: {acc_mean:0>.4f}±{acc_std:0>.4f}\t\tnmi: {nmi_mean:0>.4f}±{nmi_std:0>.4f}\t\t" \
             f"ari: {ari_mean:0>.4f}±{ari_std:0>.4f}\t\tf1: {f1_mean:0>.4f}±{f1_std:0>.4f}"
    return result


def get_format_variables(**kwargs):
    """
    get the formatted string of the input variables
    :param kwargs: variable number of variables
    :return: the formatted string with the format: variables: value\t ...
    """
    format_variables = ""
    for name, value in kwargs.items():
        format_variables += str(name) + ": " + str(value) + "\t\t"
    return format_variables


def count_parameters(model):
    """
    count the parameters' number of the input model
    Note: The unit of return value is millions(M) if exceeds 1,000,000.
    :param model: the model instance you want to count
    :return: The number of model parameters, in Million (M).
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(num_params / 1e6, 6)


def record_metrics(acc_list, nmi_list, ari_list, f1_list, metrics):
    """
    record the metrics from each iteration
    :param acc_list: the list to store acc from each iteration
    :param nmi_list: the list to store nmi from each iteration
    :param ari_list: the list to store ari from each iteration
    :param f1_list: the list to store f1 from each iteration
    :param metrics: the metrics list corresponding to the maximum acc of each iteration, including [acc, nmi, ari, f1]
    :return: acc_list, nmi_list, ari_list, f1_list
    """
    acc_list.append(metrics[0])
    nmi_list.append(metrics[1])
    ari_list.append(metrics[2])
    f1_list.append(metrics[3])
    return acc_list, nmi_list, ari_list, f1_list
