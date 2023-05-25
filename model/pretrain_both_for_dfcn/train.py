# -*- coding: utf-8 -*-
"""
@Time: 2023/5/20 16:12 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.optim import Adam

from module.AE_and_IGAE_for_DFCN import AE_IGAE
from utils.data_processor import normalize_adj, construct_graph, numpy_to_torch
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables


def train(args, data, logger):
    param_dict = {'acm': [100, 5e-5, 100],
                  'cite': [100, 1e-4, 100],
                  'cora': [100, 1e-4, 50],
                  'dblp': [100, 1e-4, 50],
                  'reut': [100, 1e-4, 100],
                  'hhar': [100, 1e-3, 50],
                  'usps': [100, 1e-3, 30]}
    args.pretrain_epoch = param_dict[args.dataset_name][0]
    args.pretrain_lr = param_dict[args.dataset_name][1]
    args.n_input = param_dict[args.dataset_name][2]
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
    args.gamma_value = 0.1
    args.lambda_value = 10
    pretrain_ae_filename = args.pretrain_ae_save_path + args.dataset_name + ".pkl"
    pretrain_igae_filename = args.pretrain_igae_save_path + args.dataset_name + ".pkl"
    logger.info("The pretraining ae .pkl file will be saved to the path: " + pretrain_ae_filename)
    logger.info("The pretraining igae .pkl file will be saved to the path: " + pretrain_igae_filename)

    model = AE_IGAE(args.dataset_name, args.ae_n_enc_1, args.ae_n_enc_2, args.ae_n_enc_3,
                    args.ae_n_dec_1, args.ae_n_dec_2, args.ae_n_dec_3,
                    args.gae_n_enc_1, args.gae_n_enc_2, args.gae_n_enc_3,
                    args.gae_n_dec_1, args.gae_n_dec_2, args.gae_n_dec_3,
                    args.n_input, args.embedding_dim, args.nodes).to(args.device)
    logger.info(model)
    model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))
    model.igae.load_state_dict(torch.load(pretrain_igae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.pretrain_lr)

    pca = PCA(n_components=args.n_input)
    X_pca = pca.fit_transform(data.feature)
    adj = data.adj.to(args.device).float()
    if args.k is not None:
        adj = construct_graph(X_pca, args.k, metric='heat').to(args.device).float()
        if args.adj_loop:
            adj = adj + torch.eye(adj.shape[0]).to(args.device).float()
        if args.adj_norm:
            adj = normalize_adj(adj, args.adj_symmetric).float()
    label = data.label
    X_pca = numpy_to_torch(X_pca).to(args.device).float()

    with torch.no_grad():
        _, _, _, z_tilde = model(X_pca, adj)

    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    acc_max, embedding = 0, 0
    max_acc_corresponding_metrics = [0, 0, 0, 0]

    for epoch in range(1, args.pretrain_epoch + 1):
        model.train()
        x_hat, adj_hat, z_hat, embedding = model(X_pca, adj)

        loss_ae = F.mse_loss(x_hat, X_pca)
        loss_w = F.mse_loss(z_hat, torch.mm(adj, X_pca))
        loss_a = F.mse_loss(adj_hat, adj)
        loss_igae = loss_w + args.gamma_value * loss_a
        loss = loss_ae + loss_igae

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(embedding.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(label, kmeans.labels_)
            if acc > acc_max:
                acc_max = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    torch.save(model.ae.state_dict(), pretrain_ae_filename)
    torch.save(model.igae.state_dict(), pretrain_igae_filename)
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    return result
