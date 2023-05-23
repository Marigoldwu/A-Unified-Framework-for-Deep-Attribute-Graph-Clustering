# -*- coding: utf-8 -*-
"""
@Time: 2023/5/8 9:06 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import argparse

parser = argparse.ArgumentParser(description='Scalable Unified Framework of Deep Graph Clustering')
# Sometimes when faced with many experimental result files, we don't remember which experiment it was.
# So, join me in forming the habit of recording the key points of the experiment!
# Describe the key points of this experiment in brief words with --desc balabala.
# You can record whatever you want!
parser.add_argument("-P", '--pretrain', dest="is_pretrain", default=False, action="store_true",
                    help="Whether to pretrain. Using '-P' to pretrain.")
parser.add_argument("-TS", "--tsne", dest="plot_clustering_tsne", default=False, action="store_true",
                    help="Whether to draw the clustering tsne image. Using '-TS' to draw clustering TSNE.")
parser.add_argument("-H", "--heatmap", dest="plot_embedding_heatmap", default=False, action="store_true",
                    help="Whether to draw the embedding heatmap. Using '-H' to draw embedding heatmap.")
parser.add_argument("-N", "--norm", dest="adj_norm", default=False, action="store_true",
                    help="Whether to normalize the adj, default is False. Using '-N' to load adj with normalization.")
parser.add_argument("-SLF", "--self_loop_false", dest="adj_loop", default=True, action="store_false",
                    help="Whether the adj has self-loop, default is True. Using '-SLF' to load adj without self-loop.")
parser.add_argument("-SF", "--symmetric_false", dest="adj_symmetric", default=True, action="store_false",
                    help="Whether the normalization type is symmetric. Using '-SF' to load asymmetric adj.")
parser.add_argument("-DS", "--desc", type=str, default="default",
                    help="The description of this experiment.")
parser.add_argument("-M", "--model", type=str, dest="model_name", default="SDCN",
                    help="The model you want to run.")
parser.add_argument("-D", '--dataset', type=str, dest="dataset_name", default="acm",
                    help="The dataset you want to use.")
parser.add_argument("-R", "--root", type=str, default=None,
                    help="Input root path to switch relative path to absolute.")
parser.add_argument("-K", "--k", type=int, default=None,
                    help="The k of KNN.")
parser.add_argument("-T", "--t", type=int, default=None,
                    help="The order in GAT. 'None' denotes don't calculate the matrix M.")
parser.add_argument("-LS", "--loops", type=int, default=1,
                    help="The Number of training rounds.")
parser.add_argument("-F", "--feature", dest="feature_type", type=str, default="tensor", choices=["tensor", "npy"],
                    help="The datatype of feature. 'tenor' and 'npy' are available.")
parser.add_argument("-L", "--label", dest="label_type", type=str, default="npy", choices=["tensor", "npy"],
                    help="The datatype of label. 'tenor' and 'npy' are available.")
parser.add_argument("-A", "--adj", dest="adj_type", type=str, default="tensor", choices=["tensor", "npy"],
                    help="The datatype of adj. 'tenor' and 'npy' are available.")
parser.add_argument("-S", "--seed", type=int, default=0,
                    help="The random seed. The default value is 0.")
