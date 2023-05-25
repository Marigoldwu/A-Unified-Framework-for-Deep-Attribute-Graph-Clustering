# -*- coding: utf-8 -*-
"""
@Time: 2023/5/23 18:19 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import numpy as np
import torch
from sklearn.decomposition import PCA

from model.DCRN.model import DCRN
from torch.optim import Adam
from sklearn.cluster import KMeans
from utils.data_processor import normalize_adj, numpy_to_torch, diffusion_adj, remove_edge, \
    gaussian_noised_feature, target_distribution
from utils.evaluation import eva
from utils.loss import reconstruction_loss, distribution_loss, dicr_loss
from utils.result import Result
from utils.utils import count_parameters, get_format_variables


def train(args, data, logger):
    param_dict = {
        "acm": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 5e-5, "n_input": 100},
        "dblp": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-4, "n_input": 50},
        "cite": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-5, "n_input": 100},
        "amap": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-3, "n_input": 100}
    }
    args.max_epoch = 400
    args.alpha_value = param_dict[args.dataset_name]["alpha_value"]
    args.lambda_value = param_dict[args.dataset_name]["lambda_value"]
    args.gamma_value = param_dict[args.dataset_name]["gamma_value"]
    args.lr = param_dict[args.dataset_name]["lr"]
    args.n_input = param_dict[args.dataset_name]["n_input"]
    args.embedding_dim = 20
    args.ae_n_enc_1 = 128
    args.ae_n_enc_2 = 256
    args.ae_n_enc_3 = 512
    args.ae_n_dec_1 = 512
    args.ae_n_dec_2 = 256
    args.ae_n_dec_3 = 128
    args.gae_n_enc_1 = 128
    args.gae_n_enc_2 = 256
    args.gae_n_enc_3 = 20
    args.gae_n_dec_1 = 20
    args.gae_n_dec_2 = 256
    args.gae_n_dec_3 = 128

    pretrain_ae_filename = args.pretrain_ae_save_path + args.dataset_name + ".pkl"
    pretrain_igae_filename = args.pretrain_igae_save_path + args.dataset_name + ".pkl"

    model = DCRN(args.dataset_name, args.ae_n_enc_1, args.ae_n_enc_2, args.ae_n_enc_3,
                 args.ae_n_dec_1, args.ae_n_dec_2, args.ae_n_dec_3,
                 args.gae_n_enc_1, args.gae_n_enc_2, args.gae_n_enc_3,
                 args.gae_n_dec_1, args.gae_n_dec_2, args.gae_n_dec_3,
                 args.n_input, args.embedding_dim, args.clusters, n_node=args.nodes).to(args.device)
    logger.info(model)
    model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))
    model.igae.load_state_dict(torch.load(pretrain_igae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr)

    pca = PCA(n_components=args.n_input)
    X_pca = pca.fit_transform(data.feature)
    X_pca = numpy_to_torch(X_pca).to(args.device).float()
    adj = data.adj + np.eye(data.adj.shape[0])
    adj_norm = normalize_adj(adj, symmetry=False)
    adj_norm = numpy_to_torch(adj_norm).to(args.device).float()
    Ad = diffusion_adj(data.adj, mode='ppr', transport_rate=args.alpha_value)
    Ad = numpy_to_torch(Ad).to(args.device).float()
    label = data.label
    adj = numpy_to_torch(adj)

    with torch.no_grad():
        _, _, _, sim, _, _, _, Z, _, _ = model(X_pca, adj_norm, X_pca, adj_norm)

    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(Z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    Am = remove_edge(adj, sim, remove_rate=0.1, device=args.device).float()

    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        # add gaussian noise to X
        X_tilde1, X_tilde2 = gaussian_noised_feature(X_pca, args.device)
        X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, embedding, AZ_all, Z_all = model(X_tilde1, Ad, X_tilde2, Am)

        # calculate loss: L_{DICR}, L_{REC} and L_{KL}
        L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all, args.dataset_name, args.gamma_value)
        L_REC = reconstruction_loss(X_pca, adj_norm, X_hat, Z_hat, A_hat)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss = L_DICR + L_REC + args.lambda_value * L_KL

        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(embedding.data.cpu().numpy())
        acc, nmi, ari, f1 = eva(label, kmeans.labels_)
        if acc > acc_max:
            acc_max = acc
            max_acc_corresponding_metrics = [acc, nmi, ari, f1]
        logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                         ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The max memory allocated to model is: {mem_used:.2f} MB.")
    return result
