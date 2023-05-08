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
# You can record whatever you want to!
parser.add_argument("-D", "--desc", type=str, default="default",
                    help="description of this experiment")
parser.add_argument("-M", "--model", type=str, dest="model_name", default="SDCN",
                    help="the model you want to run")
parser.add_argument("-S", '--dataset', type=str, dest="dataset_name", default="acm",
                    help="the dataset you want to use")
parser.add_argument("-R", "--root", type=str, default=None,
                    help="input root path to switch relative path to absolute")
parser.add_argument("-P", '--pretrain', dest="is_pretrain", default=False, action="store_true",
                    help="is it pretrain")
parser.add_argument("-C", "--tsne", dest="plot_clustering_tsne", default=False, action="store_true",
                    help="whether to draw the clustering tsne image")
parser.add_argument("-H", "--heatmap", dest="plot_embedding_heatmap", default=False, action="store_true",
                    help="whether to draw the embedding heatmap")
parser.add_argument("-K", "--k", type=int, default=None,
                    help="the k of KNN")
parser.add_argument("-T", "--t", type=int, default=2,
                    help="the order in GAT")
parser.add_argument("-L", "--loops", type=int, default=1,
                    help="number of training rounds")
