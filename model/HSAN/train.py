# -*- coding: utf-8 -*-
"""
@Time: 2023/5/22 18:35 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.optim import Adam
from model.HSAN.model import HardSampleAwareNetwork
from utils.data_processor import laplacian_filtering, comprehensive_similarity, hard_sample_aware_infoNCE, \
    high_confidence, pseudo_matrix, numpy_to_torch
from utils.evaluation import eva
from utils.kmeans_gpu import kmeans
from utils.utils import get_format_variables, count_parameters


def train(args, data, logger):
    param_dict = {
        "cora": {"lap_t": 2, "lr": 1e-3, "n_input": 500, "dims": 1500, "activate": 'ident', "tao": 0.9, "beta": 1},
        "cite": {"lap_t": 2, "lr": 1e-3, "n_input": 500, "dims": 1500, "activate": 'sigmoid', "tao": 0.3, "beta": 2},
        "amap": {"lap_t": 3, "lr": 1e-5, "n_input": -1, "dims": 500, "activate": 'ident', "tao": 0.9, "beta": 3},
        "bat": {"lap_t": 6, "lr": 1e-3, "n_input": -1, "dims": 1500, "activate": 'ident', "tao": 0.3, "beta": 5},
        "eat": {"lap_t": 6, "lr": 1e-4, "n_input": -1, "dims": 1500, "activate": 'ident', "tao": 0.7, "beta": 5},
        "uat": {"lap_t": 6, "lr": 1e-4, "n_input": -1, "dims": 500, "activate": 'sigmoid', "tao": 0.7, "beta": 5},
    }
    if args.dataset_name in param_dict:
        args.lap_t = param_dict[args.dataset_name]["lap_t"]
        args.lr = param_dict[args.dataset_name]["lr"]
        args.n_input = param_dict[args.dataset_name]["n_input"]
        args.dims = param_dict[args.dataset_name]["dims"]
        args.activate = param_dict[args.dataset_name]["activate"]
        args.tao = param_dict[args.dataset_name]["tao"]
        args.beta = param_dict[args.dataset_name]["beta"]
    else:
        args.lap_t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 1
    if args.n_input != -1:
        args.input_dim = args.n_input
    args.max_epoch = 400
    model = HardSampleAwareNetwork(args.input_dim, args.dims, args.activate, args.nodes).to(args.device)
    logger.info(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    label = data.label
    feature = data.feature
    if args.n_input != -1:
        pca = PCA(n_components=args.n_input)
        feature = pca.fit_transform(feature)

    # apply the laplacian filtering
    X_filtered = laplacian_filtering(data.adj.astype(np.float64),
                                     feature.astype(np.float64), args.lap_t).to(args.device)
    adj = numpy_to_torch(data.adj).to(args.device).float()

    # positive and negative sample pair index matrix
    mask = torch.ones([args.nodes * 2, args.nodes * 2]) - torch.eye(args.nodes * 2)
    mask = mask.to(args.device)

    acc_max, Z = 0, 0
    acc_max_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(args.max_epoch):
        model.train()
        # encoding with Eq. (3)-(5)
        Z1, Z2, E1, E2 = model(X_filtered, adj)

        # calculate comprehensive similarity by Eq. (6)
        S = comprehensive_similarity(Z1, Z2, E1, E2, model.alpha)

        # calculate hard sample aware contrastive loss by Eq. (10)-(11)
        loss = hard_sample_aware_infoNCE(S, mask, model.pos_neg_weight, model.pos_weight, args.nodes)

        # optimization
        loss.backward()
        optimizer.step()
        # testing and update weights of sample pairs
        if epoch % 10 == 0:
            # evaluation mode
            model.eval()

            # encoding
            Z1, Z2, E1, E2 = model(X_filtered, adj)

            # calculate comprehensive similarity by Eq. (6)
            S = comprehensive_similarity(Z1, Z2, E1, E2, model.alpha)

            # fusion and testing
            Z = (Z1 + Z2) / 2
            predict_labels, centers = kmeans(X=Z, num_clusters=args.clusters, distance="euclidean", device="cuda")
            P = predict_labels.numpy()
            acc, nmi, ari, f1 = eva(label, P)

            # select high confidence samples
            H, H_mat = high_confidence(Z, centers, args.tao)

            # calculate new weight of sample pair by Eq. (9)
            M, M_mat = pseudo_matrix(P, S, args.nodes, args.beta, args.device)

            # update weight
            model.pos_weight[H] = M[H].data
            model.pos_neg_weight[H_mat] = M_mat[H_mat].data

            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The total memory allocated to model is: {mem_used:.2f} MB.")
    return Z, acc_max_corresponding_metrics
