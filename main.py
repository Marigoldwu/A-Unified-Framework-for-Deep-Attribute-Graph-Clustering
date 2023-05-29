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

from utils.options import parser
from dataset import dataset_info
from utils import logger, time_manager, path_manager, plot, rand
from utils.load_data import load_data
from utils.utils import cal_mean_std, record_metrics


if __name__ == "__main__":
    args = parser.parse_args()
    # setup random seed to ensure that the experiment can be reproduced
    rand.setup_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # get the information of dataset
    args = dataset_info.get_dataset_info(args)
    # get the relative path or absolute path
    args = path_manager.get_path(args)

    # Configuration of logger and timer module.
    # The logger print the training log to the specified file and the timer record training's time assuming.
    logger = logger.MyLogger(args.model_name, log_file_path=f"{args.log_save_path}{time_manager.get_format_time()}.log")
    logger.info("The key points of this experiment: " + args.desc)
    logger.info(f"random seed: {args.seed}")
    timer = time_manager.MyTime()

    # Load data, including features, label, adjacency matrix.
    data = load_data(args.k, args.dataset_path, args.dataset_name,
                     feature_type=args.feature_type, label_type=args.label_type, adj_type=args.adj_type,
                     adj_loop=args.adj_loop, adj_norm=args.adj_norm, adj_symmetric=args.adj_symmetric,
                     t=args.t)

    # Auto import the training module of the model you specified.
    model_train = importlib.import_module(f"model.{args.model_name}.train")
    train = getattr(model_train, "train")

    # Training
    acc_list, nmi_list, ari_list, f1_list = [], [], [], []
    # repeat args.loops rounds
    for i in range(args.loops):
        logger.info(f"{'=' * 20}Training loop No.{i + 1}{'=' * 20}")
        timer.start()
        # call the training function of your specified model
        result = train(args, data, logger)

        seconds, minutes = timer.stop()
        logger.info("Time consuming: {}s or {}m".format(seconds, minutes))

        # record the max value of each loop
        acc_list, nmi_list, ari_list, f1_list = record_metrics(acc_list, nmi_list, ari_list, f1_list,
                                                               result.max_acc_corresponding_metrics)
        # draw the clustering image or embedding heatmap
        if args.plot_clustering_tsne:
            plot.plot_clustering_tsne(args, result.embedding, data.label,
                                      logger, desc=f"{i}", title=None, axis_show=False)
        if args.plot_embedding_heatmap:
            plot.plot_embedding_heatmap(args, torch.matmul(result.embedding, result.embedding.t()),
                                        logger, desc=f"{i}", title=None,
                                        axis_show=False, color_bar_show=True)

    logger.info(str(args))
    logger.info("Total loops: {}".format(args.loops))
    logger.info("Mean value:")
    logger.info(cal_mean_std(acc_list, nmi_list, ari_list, f1_list))
    logger.info("Training over! Punch out!")
