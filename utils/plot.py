# -*- coding: utf-8 -*-
"""
@Time: 2023/4/28 18:09 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_clustering_tsne(args, embedding, label, logger, img_suffix=".PDF",
                         axis_show=True, title="TSNE", legend_show=True):
    """
    :param args: the parameter settings of model
    :param embedding: the embedded representations will be drawn which were learned by model
    :param label: the groundtruth label of dataset
    :param logger: the logger to record information during the process
    :param img_suffix: the suffix of image
            '.png','.pdf','.jpg','.jpeg','.bmp','.tiff','.gif','.svg', '.eps' are available
    :param axis_show: is show the axis of image
    :param title: the title of image, default value is "TSNE", if needn't, set it to None
    :param legend_show: whether to display the legend of the image, default value is True
    :return:
    """
    support_suffix = [".png", ".pdf", ".jpg", "jpeg", ".bmp", ".tiff", ".gif", ".svg", ".eps"]
    if support_suffix not in support_suffix:
        logger.error("The suffix is not supported!")
        return
    clustering_tsne_filename = args.clustering_tsne_save_path + args.dataset_name + img_suffix
    logger.info("==========正在绘图==========")
    X_tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embedding.cpu().detach().numpy())
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, c=label)
    if not axis_show:
        plt.axis("off")
    if title is not None:
        plt.title(title)
    if legend_show:
        plt.legend()
    plt.savefig(clustering_tsne_filename)
    logger.info("==========绘图结束==========")
    logger.info("The .pdf image was saved to: " + clustering_tsne_filename)
