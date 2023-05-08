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


def validate_suffix(suffix):
    support_suffix = [".png", ".pdf", ".jpg", "jpeg", ".bmp", ".tiff", ".gif", ".svg", ".eps"]
    if suffix not in support_suffix:
        return False
    return True


def plot_clustering_tsne(args, embedding, label, logger, img_suffix=".pdf", desc="default",
                         axis_show=True, title="TSNE"):
    """
    :param desc: the description of image
    :param args: the parameter settings of model
    :param embedding: the embedded representations will be drawn which were learned by model
    :param label: the groundtruth label of dataset
    :param logger: the logger to record information during the process
    :param img_suffix: the suffix of image
            '.png','.pdf','.jpg','.jpeg','.bmp','.tiff','.gif','.svg', '.eps' are available
    :param axis_show: is show the axis of image
    :param title: the title of image, default value is "TSNE", if needn't, set it to None
    :return:
    """
    if not validate_suffix(img_suffix):
        logger.error("The suffix is not supported! Skip drawing!")
        return
    clustering_tsne_filename = f"{args.clustering_tsne_save_path}{args.dataset_name}_{desc}{img_suffix}"
    logger.info("==========Start Drawing TSNE==========")
    X_tsne = TSNE(n_components=2).fit_transform(embedding.cpu().detach().numpy())
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, c=label)
    if not axis_show:
        plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(clustering_tsne_filename)
    logger.info("===========End Drawing TSNE===========")
    logger.info("The clustering tsne visualization image was saved to: " + clustering_tsne_filename)
    plt.clf()


def plot_embedding_heatmap(args, embedding, logger, img_suffix=".pdf", desc="default",
                           axis_show=True, title="Heatmap", color_bar_show=True):
    """
    :param desc: the description of image
    :param args: the parameter settings of model
    :param embedding: the embedded representations will be drawn which were learned by model
    :param logger: the logger to record information during the process
    :param img_suffix: the suffix of image
            '.png','.pdf','.jpg','.jpeg','.bmp','.tiff','.gif','.svg', '.eps' are available
    :param axis_show: is show the axis of image
    :param title: the title of image, default value is "HeatMap", if needn't, set it to None
    :param color_bar_show: whether to display the color bar of the image, default value is True
    :return:
    """
    if not validate_suffix(img_suffix):
        logger.error("The suffix is not supported! Skip drawing!")
        return
    embedding_heatmap_filename = f"{args.embedding_heatmap_save_path}{args.dataset_name}_{desc}{img_suffix}"
    logger.info("==========Start Drawing Heatmap==========")
    plt.imshow(embedding.cpu().detach().numpy(), cmap=plt.cm.GnBu, interpolation='nearest')
    if color_bar_show:
        plt.colorbar()
    if not axis_show:
        plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(embedding_heatmap_filename)
    logger.info("===========End Drawing Heatmap===========")
    logger.info("The embedding heatmap image was saved to: " + embedding_heatmap_filename)
    plt.clf()
