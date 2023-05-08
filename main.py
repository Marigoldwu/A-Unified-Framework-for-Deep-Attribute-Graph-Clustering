# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 11:16 
@Author: Marigold
@Version: 0.0.0
@Description：the entrance file of deep graph clustering
@WeChat Account: Marigold
"""
import torch
import importlib
import numpy as np

from utils.options import parser
from dataset import dataset_info
from utils import load_data, logger, time_manager, path_manager, calculator, plot, rand, data_processor
from utils.record import record_metrics

if __name__ == "__main__":
    rand.setup_seed(325)
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args = dataset_info.get_dataset_info(args)
    args = path_manager.get_path(args)

    # Configuration of logger and timer module.
    # The logger print the training log to the specified file and the timer record training's time assuming.
    logger_file_name = f"{time_manager.get_format_time()}.log"
    logger = logger.MyLogger(args.model_name, log_file_path=args.log_save_path + logger_file_name)
    logger.info("The key points of this experiment: " + args.desc)
    logger.info(str(args))

    timer = time_manager.MyTime()

    # Load data, including features, label, adjacency matrix.
    # If the dataset is not graph dataset, use construct_graph to get KNN graph.
    # Note: the store format of data is numpy, and the adj has no self-loop.
    if args.k is None:
        feature, label, adj = load_data.load_graph_data(args.dataset_path, args.dataset_name)
    else:
        feature, label = load_data.load_data(args.dataset_path, args.dataset_name)
        metric_dict = {"usps": "heat", "hhar": "cosine", "reut": "cosine"}
        adj = data_processor.construct_graph(feature, args.k, metric_dict[args.dataset_name])

    # Auto import the training module of the model you specified.
    model_train = importlib.import_module(f"model.{args.model_name}.train")
    train = getattr(model_train, "train")

    # Training
    acc_list, nmi_list, ari_list, f1_list = [], [], [], []
    # repeat args.loops rounds
    for i in range(args.loops):
        logger.info(f"==================Training loop No.{i+1}==================")
        timer.start()
        # call the training function of your specified model
        embedding, max_acc_corresponding_metrics = train(args, feature, label, adj, logger)

        seconds, minutes = timer.stop()
        logger.info("Time consuming: {}s or {}m".format(seconds, minutes))

        # record the max value of each loop
        acc_list, nmi_list, ari_list, f1_list = record_metrics(acc_list, nmi_list, ari_list, f1_list,
                                                               max_acc_corresponding_metrics)
        # draw the clustering image or embedding heatmap
        if args.plot_clustering_tsne:
            plot.plot_clustering_tsne(args, embedding, label, logger, desc=f"{i}", title=None, axis_show=False)
        if args.plot_embedding_heatmap:
            plot.plot_embedding_heatmap(args, embedding, logger, desc=f"{i}", title=None,
                                        axis_show=False, color_bar_show=True)

    logger.info("Total loops: {}".format(args.loops))
    logger.info("Mean value:")
    logger.info(calculator.cal_mean_std(acc_list, nmi_list, ari_list, f1_list))
    logger.info("Training over! Punch out!")
