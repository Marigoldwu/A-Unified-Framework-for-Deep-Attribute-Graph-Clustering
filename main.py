# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 11:16 
@Author: Marigold
@Version: 0.0.0
@Description：the entrance file of deep graph clustering
@WeChat Account: Marigold
"""
import importlib
import torch
import argparse
from utils import load_data, logger, time_manager, path_manager, calculator, plot
from dataset import dataset_info

if __name__ == "__main__":
    # Sometimes when faced with many experimental result files, we don't remember which experiment it was.
    # So, join me in forming the habit of recording the key points of the experiment!
    # Describe the key points of this experiment in brief words. You can record whatever you want to!
    description = "统计参数量和模型耗时"
    root_path = "/content/Drive/MyDrive"
    """
    long model_name for copy conveniently:
    pretrain_model:
        pretrain_ae:
            - pretrain_ae_for_sdcn
        pretrain_gat:
            - pretrain_gat_for_daegc
    """
    parser = argparse.ArgumentParser(description='Unified Code Framework of Deep Graph Clustering')
    parser.add_argument('--is_pretrain', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default="SDCN")
    parser.add_argument('--dataset_name', type=str, default="acm")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--update_interval", type=int, default=1)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--is_change_root_path", type=bool, default=False)
    parser.add_argument("--plot_clustering_tsne", type=bool, default=False)
    parser.add_argument("--plot_embedding_heatmap", type=bool, default=False)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args = dataset_info.get_dataset_info(args.dataset_name, args)
    args = path_manager.get_abs_path(args, root_path)

    # Configuration of logger and timer module.
    # The logger print the training log to the specified file and the timer record training's time assuming.
    current_time = time_manager.get_format_time()
    logger_file_name = current_time + ".log"
    logger = logger.MyLogger(args.model_name, log_file_path=args.log_save_path + logger_file_name)
    logger.info("The key points of this experiment: " + description)
    logger.info(str(args))

    timer = time_manager.MyTime()

    # Load data, including features, label, adjacency matrix.
    # Note: the store format of data is numpy.
    feature, label, adj = load_data.load_graph_data(args.dataset_path, args.dataset_name)

    # Auto import the training module of the model you specified.
    model_train = importlib.import_module(f"model.{args.model_name}.train")
    train = getattr(model_train, "train")

    # Training
    acc, nmi, ari, f1 = [], [], [], []
    for i in range(args.loops):
        logger.info("Training loop No.{}".format(i + 1))
        timer.start()
        embedding, max_acc_corresponding_metrics = train(args, feature, label, adj, logger)
        acc.append(max_acc_corresponding_metrics[0])
        nmi.append(max_acc_corresponding_metrics[1])
        ari.append(max_acc_corresponding_metrics[2])
        f1.append(max_acc_corresponding_metrics[3])
        if args.plot_clustering_tsne:
            plot.plot_clustering_tsne(args, embedding, label, logger)
        if args.plot_embedding_heatmap:
            plot.plot_embedding_heatmap(args, embedding, logger)
        seconds, minutes = timer.stop()
        logger.info("Time assuming: {}s or {}m".format(seconds, minutes))

    logger.info("Total loops: {}".format(args.loops))
    logger.info("Mean value:")
    logger.info(calculator.cal_mean_std(acc, nmi, ari, f1))
    logger.info("Training over! Punch out!")
