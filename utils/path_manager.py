# -*- coding: utf-8 -*-
"""
@Time: 2023/4/28 9:11 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import os


def replace_relative_path(relative_path, root_path):
    return relative_path.replace(".", root_path, 1)


def get_path(args):
    """
    Get the path. If root is not None and valid, it will switch relative path to absolute according to the root.
    Include:
    The path where pretrain file will be saved to, such as pretrain_ae_save_path.
    The path where log file will be saved to, such as log_save_path.
    The path where dataset will be loaded, such as dataset_path.
    The path where clustering tsne image will be saved to, such as clustering_tsne_save_path.
    The path where embedding heatmap image will be saved to, such as embedding_heatmap_save_path.
    :param args: argparse object
    :return: Assign path to args and return args
    """
    pretrain_type_list = []
    # If is pretraining, we can get the pretrain type from model name directly. A big problem is that the type of
    # model pretraining is uncertain, so we should tell the program what pretraining types the model has. Here we use a
    # pretrain_type_dict to store the pretraining types. So, if you want to run your own model, you need to add the
    # model name here first, otherwise an error will be reported.
    if args.is_pretrain:
        model_name_list = args.model_name.split("_")
        pretrain_type = model_name_list[0] + "_" + model_name_list[1]
        pretrain_for = (model_name_list[-1]).upper()
    else:
        pretrain_type_dict = {"DAEGC": ["pretrain_gat"],
                              "SDCN": ["pretrain_ae"],
                              "AGCN": ["pretrain_ae"],
                              "EFRDGC": ["pretrain_ae", "pretrain_gat"],
                              "GCSEE": ["pretrain_ae", "pretrain_gat"],
                              "GCAE": ["pretrain_gae"],
                              "FDAEGC": ["pretrain_fgat"],
                              "RSGC": ["pretrain_gat"],
                              "DFCN": ["pretrain_ae", "pretrain_igae"],
                              "HSAN": []}
        pretrain_for = args.model_name
        pretrain_type_list = pretrain_type_dict[args.model_name]
        if len(pretrain_type_list) == 1:
            pretrain_type = pretrain_type_list[0]
        elif len(pretrain_type_list) == 0:
            pretrain_type = None
        else:
            pretrain_type = "multi"
            for item in pretrain_type_list:
                type_name = item.split("_")[-1]
                exec(f"args.pretrain_{type_name}_save_path = "
                     f"'./pretrain/pretrain_{type_name}/{pretrain_for}/{args.dataset_name}/'")
    directory_structure = args.model_name + "/" + args.dataset_name + "/"
    args.log_save_path = "./logs/" + directory_structure
    args.dataset_path = "./"
    args.clustering_tsne_save_path = "./img/clustering/" + directory_structure
    args.embedding_heatmap_save_path = "./img/heatmap/" + directory_structure

    if pretrain_type == "pretrain_ae":
        args.pretrain_save_path = "./pretrain/pretrain_ae/" + pretrain_for + "/" + args.dataset_name + "/"
    elif pretrain_type == "pretrain_gae":
        args.pretrain_save_path = "./pretrain/pretrain_gae/" + pretrain_for + "/" + args.dataset_name + "/"
    elif pretrain_type == "pretrain_igae":
        args.pretrain_save_path = "./pretrain/pretrain_igae/" + pretrain_for + "/" + args.dataset_name + "/"
    elif pretrain_type == "pretrain_gat":
        args.pretrain_save_path = "./pretrain/pretrain_gat/" + pretrain_for + "/" + args.dataset_name + "/"
    elif pretrain_type == "pretrain_igat":
        args.pretrain_save_path = "./pretrain/pretrain_igat/" + pretrain_for + "/" + args.dataset_name + "/"
    elif pretrain_type == "pretrain_fgat":
        args.pretrain_save_path = "./pretrain/pretrain_fgat/" + pretrain_for + "/" + args.dataset_name + "/"
    elif pretrain_type == "pretrain_both":
        args.pretrain_ae_save_path = "./pretrain/pretrain_ae/" + pretrain_for + "/" + args.dataset_name + "/"
        args.pretrain_igae_save_path = "./pretrain/pretrain_igae/" + pretrain_for + "/" + args.dataset_name + "/"
    elif pretrain_type is None:
        pass
    elif pretrain_type == "multi":
        pass
    else:
        print("The pretraining type error!"
              "Please check the pretrain type name or complete the if-elif above with your type!")
    root = args.root
    if root is not None:
        if not os.path.exists(root):
            raise FileNotFoundError(f"{root} not Found!")
        args.log_save_path = replace_relative_path(args.log_save_path, root)
        args.dataset_path = replace_relative_path(args.dataset_path, root)
        args.clustering_tsne_save_path = replace_relative_path(args.clustering_tsne_save_path, root)
        args.embedding_heatmap_save_path = replace_relative_path(args.embedding_heatmap_save_path, root)
        if pretrain_type == "multi":
            for item in pretrain_type_list:
                type_name = item.split("_")[-1]
                exec(f"args.pretrain_{type_name}_save_path = "
                     f"replace_relative_path(args.pretrain_{type_name}_save_path, root_path)")
        elif pretrain_type is not None:
            args.pretrain_save_path = replace_relative_path(args.pretrain_save_path, root)
    if not os.path.exists(args.log_save_path):
        os.makedirs(args.log_save_path)
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"{args.dataset_path} not found!")
    if args.plot_clustering_tsne and not os.path.exists(args.clustering_tsne_save_path):
        os.makedirs(args.clustering_tsne_save_path)
    if args.plot_embedding_heatmap and not os.path.exists(args.embedding_heatmap_save_path):
        os.makedirs(args.embedding_heatmap_save_path)
    if pretrain_type == "multi":
        for item in pretrain_type_list:
            type_name = item.split("_")[-1]
            path = getattr(args, f"pretrain_{type_name}_save_path")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found!")
    elif pretrain_type == "pretrain_both":
        pass
    elif pretrain_type is not None:
        if not os.path.exists(args.pretrain_save_path):
            os.makedirs(args.pretrain_save_path)
    return args
