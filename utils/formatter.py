# -*- coding: utf-8 -*-
"""
@Time: 2023/4/28 8:45 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""


def get_format_variables(**kwargs):
    format_variables = ""
    for name, value in kwargs.items():
        format_variables += str(name) + ": " + str(value) + "\t\t"
    return format_variables
