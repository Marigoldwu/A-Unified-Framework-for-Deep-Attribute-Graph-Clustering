# -*- coding: utf-8 -*-
"""
@Time: 2023/4/29 16:54 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch
import torch.nn.functional as F

from torch.optim import Adam
from model.AGCN.model import AGCN
from sklearn.cluster import KMeans
from utils import data_processor
from utils.data_processor import target_distribution
from utils.evaluation import eva
from utils.utils import count_parameters, get_format_variables


def train(args, feature, label, adj, logger):
    args.embedding_dim = 10
    args.enc_1_dim = 500
    args.enc_2_dim = 500
    args.enc_3_dim = 2000
    args.dec_1_dim = 2000
    args.dec_2_dim = 500
    args.dec_3_dim = 500
    args.max_epoch = 50

    # hyper parameters
    lambda_dict = {"usps": [1000, 1000], "hhar": [1, 0.1], "reut": [10, 10]}
    lambda_1 = 0.1 if args.k is None else lambda_dict[args.dataset_name][0]
    lambda_2 = 0.01 if args.k is None else lambda_dict[args.dataset_name][1]
    logger.info(get_format_variables(lambda_1=lambda_1, lambda_2=lambda_2))

    pretrain_ae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"

    model = AGCN(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                 enc_1_dim=args.enc_1_dim, enc_2_dim=args.enc_2_dim, enc_3_dim=args.enc_3_dim,
                 dec_1_dim=args.dec_1_dim, dec_2_dim=args.dec_2_dim, dec_3_dim=args.dec_3_dim,
                 clusters=args.clusters, v=1.0).to(args.device)
    logger.info(model)

    model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr)
    # cluster parameter initiate

    feature = data_processor.numpy_to_torch(feature).to(args.device).float()
    adj = data_processor.normalize_adj(adj)
    adj = data_processor.numpy_to_torch(adj).to(args.device).float()

    with torch.no_grad():
        _, _, _, _, z = model.ae(feature)

    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

    acc_max, embedding = 0, 0
    acc_max_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        x_bar, q, pred, _, embedding = model(feature, adj)
        p = target_distribution(q.data)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, feature)
        loss = lambda_1 * kl_loss + lambda_2 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            _, _, pred, _, _ = model(feature, adj)
            y_pred = pred.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, y_pred)
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The total memory allocated to model is: {mem_used:.2f} MB.")
    return embedding, acc_max_corresponding_metrics
