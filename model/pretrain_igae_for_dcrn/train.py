# -*- coding: utf-8 -*-
"""
@Time: 2023/5/23 19:32 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.decomposition import PCA
from torch.optim import Adam
from sklearn.cluster import KMeans
from module.IGAE import IGAE
from utils.data_processor import numpy_to_torch
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables


def train(args, data, logger):
    param_dict = {
        "acm": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-3, "n_input": 100},
        "dblp": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-3, "n_input": 50},
        "cite": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-3, "n_input": 100},
        "amap": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-3, "n_input": 100}
    }
    args.pretrain_epoch = 30
    args.pretrain_lr = param_dict[args.dataset_name]["lr"]
    args.n_input = param_dict[args.dataset_name]["n_input"]
    args.gamma_value = param_dict[args.dataset_name]["gamma_value"]
    args.gae_n_enc_1 = 128
    args.gae_n_enc_2 = 256
    args.gae_n_enc_3 = 20
    args.gae_n_dec_1 = 20
    args.gae_n_dec_2 = 256
    args.gae_n_dec_3 = 128
    pretrain_igae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    logger.info("The pretrain .pkl file will be saved to the path: " + pretrain_igae_filename)
    model = IGAE(args.dataset_name, args.gae_n_enc_1, args.gae_n_enc_2, args.gae_n_enc_3,
                 args.gae_n_dec_1, args.gae_n_dec_2, args.gae_n_dec_3, args.n_input).to(args.device)
    logger.info(model)
    optimizer = Adam(model.parameters(), lr=args.pretrain_lr)

    pca = PCA(n_components=args.n_input)
    X_pca = pca.fit_transform(data.feature)
    adj = data.adj.to(args.device).float()
    label = data.label

    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    X_pca = numpy_to_torch(X_pca).to(args.device).float()
    for epoch in range(1, args.pretrain_epoch+1):
        model.train()
        embedding, z_hat, adj_hat = model(X_pca, adj)
        loss_w = F.mse_loss(z_hat, torch.mm(adj, X_pca))
        loss_a = F.mse_loss(adj_hat, adj)
        loss = loss_w + args.gamma_value * loss_a
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
                max_embedding = embedding
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    torch.save(model.state_dict(), pretrain_igae_filename)
    # Sort F based on the sort indices
    sort_indices = np.argsort(data.label)
    max_embedding = max_embedding[sort_indices]
    result = Result(embedding=max_embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    return result
