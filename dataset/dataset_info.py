# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 20:02 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""


def get_dataset_info(args):
    name = args.dataset_name
    if name == "acm":
        args.clusters = 3
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1870
    elif name == "cite":
        args.clusters = 6
        args.lr = 1e-4
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 3703
    elif name == "dblp":
        args.clusters = 4
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 334
    elif name == "cora":
        args.clusters = 7
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1433
    elif name == "pubmed":
        args.clusters = 3
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 500
    elif name == "wiki":
        args.clusters = 17
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 4973
    elif name == "eat":
        args.clusters = 4
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 203
    elif name == "bat":
        args.clusters = 4
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 81
    elif name == "uat":
        args.clusters = 4
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 239
    elif name == "film":
        args.clusters = 5
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 932
    elif name == "wisc":
        args.clusters = 5
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "texas":
        args.clusters = 5
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "cornell":
        args.clusters = 5
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "cocs":
        args.clusters = 5
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "corafull":
        args.clusters = 70
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 8710
    elif name == "amac":
        args.clusters = 10
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 767
    elif name == "amap":
        args.clusters = 8
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 745
    elif name == "hhar":
        args.k = 3
        args.clusters = 6
        args.lr = 1e-3
        args.max_epoch = 200
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 561
    elif name == "reut":
        args.k = 3
        args.clusters = 4
        args.lr = 1e-4
        args.max_epoch = 200
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 2000
    elif name == "usps":
        args.k = 3
        args.clusters = 10
        args.lr = 1e-3
        args.max_epoch = 200
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 256
    return args
