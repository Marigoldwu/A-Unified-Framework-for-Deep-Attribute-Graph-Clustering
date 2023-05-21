# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 14:53 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from model.DFCN.model import DFCN
from torch.optim import Adam
from sklearn.cluster import KMeans
from utils import data_processor
from utils.data_processor import construct_graph, normalize_adj, numpy_to_torch
from utils.evaluation import eva
from utils.utils import count_parameters, get_format_variables


def train(args, data, logger):
    param_dict = {'acm': [200, 5e-5, 100],
                  'cite': [200, 1e-4, 100],
                  'cora': [200, 1e-4, 50],
                  'dblp': [200, 1e-4, 50],
                  'reut': [200, 1e-4, 100],
                  'hhar': [200, 1e-3, 50],
                  'usps': [200, 1e-3, 30]}
    args.max_epoch = param_dict[args.dataset_name][0]
    args.lr = param_dict[args.dataset_name][1]
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

    model = DFCN(args.dataset_name, args.ae_n_enc_1, args.ae_n_enc_2, args.ae_n_enc_3,
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
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model(X_pca, adj)

    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    acc_max, embedding = 0, 0
    acc_max_corresponding_metrics = []
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde = model(X_pca, adj)
        tmp_q = q.data
        p = data_processor.target_distribution(tmp_q)

        loss_ae = F.mse_loss(x_hat, X_pca)
        loss_w = F.mse_loss(z_hat, torch.mm(adj, X_pca))
        loss_a = F.mse_loss(adj_hat, adj)
        loss_igae = loss_w + args.gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + args.lambda_value * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

        acc, nmi, ari, f1 = eva(label, kmeans.labels_)
        if acc > acc_max:
            acc_max = acc
            acc_max_corresponding_metrics = [acc, nmi, ari, f1]
        logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                         ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The max memory allocated to model is: {mem_used:.2f} MB.")
    return embedding, acc_max_corresponding_metrics
