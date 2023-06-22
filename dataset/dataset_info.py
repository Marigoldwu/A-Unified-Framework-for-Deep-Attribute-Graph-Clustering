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
        args.nodes = 3025
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1870
    elif name == "cite":
        args.clusters = 6
        args.nodes = 3327
        args.lr = 1e-4
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 3703
    elif name == "dblp":
        args.clusters = 4
        args.nodes = 4057
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 334
    elif name == "cora":
        args.clusters = 7
        args.nodes = 2708
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1433
    elif name == "pubmed":
        args.clusters = 3
        args.nodes = 19717
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 500
    elif name == "wiki":
        args.clusters = 17
        args.nodes = 2405
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 4973
    elif name == "eat":
        args.clusters = 4
        args.nodes = 399
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 203
    elif name == "bat":
        args.clusters = 4
        args.nodes = 131
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 81
    elif name == "uat":
        args.clusters = 4
        args.nodes = 1190
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 239
    elif name == "film":
        args.clusters = 5
        args.nodes = 7600
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 932
    elif name == "wisc":
        args.clusters = 5
        args.nodes = 251
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "texas":
        args.clusters = 5
        args.nodes = 183
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "cornell":
        args.clusters = 5
        args.nodes = 183
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "cocs":
        args.clusters = 5
        args.nodes = 18333
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 1703
    elif name == "corafull":
        args.clusters = 70
        args.nodes = 19793
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 8710
    elif name == "amac":
        args.clusters = 10
        args.nodes = 13752
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 767
    elif name == "amap":
        args.clusters = 8
        args.nodes = 7650
        args.lr = 1e-3
        args.max_epoch = 50
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 745
    elif name == "hhar":
        args.k = 3
        args.nodes = 10299
        args.clusters = 6
        args.lr = 1e-3
        args.max_epoch = 200
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 561
    elif name == "reut":
        args.k = 3
        args.nodes = 10000
        args.clusters = 4
        args.lr = 1e-4
        args.max_epoch = 200
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 2000
    elif name == "usps":
        args.k = 3
        args.nodes = 9298
        args.clusters = 10
        args.lr = 1e-3
        args.max_epoch = 200
        args.pretrain_lr = 1e-3
        args.pretrain_epoch = 30
        args.input_dim = 256
    return args
