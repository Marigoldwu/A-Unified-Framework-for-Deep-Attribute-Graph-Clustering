# -*- coding: utf-8 -*-
"""
@Time: 2023/4/28 16:28 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F

from model.SDCN.model import SDCN
from torch.optim import Adam
from sklearn.cluster import KMeans
from utils import data_processor
from utils.evaluation import eva
from utils.utils import count_parameters, get_format_variables


def train(args, data, logger):
    args.embedding_dim = 10
    args.enc_1_dim = 500
    args.enc_2_dim = 500
    args.enc_3_dim = 2000
    args.dec_1_dim = 2000
    args.dec_2_dim = 500
    args.dec_3_dim = 500
    pretrain_ae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"

    model = SDCN(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                 enc_1_dim=args.enc_1_dim, enc_2_dim=args.enc_2_dim, enc_3_dim=args.enc_3_dim,
                 dec_1_dim=args.dec_1_dim, dec_2_dim=args.dec_2_dim, dec_3_dim=args.dec_3_dim,
                 clusters=args.clusters,
                 v=1).to(args.device)
    logger.info(model)

    model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr)

    feature = data.feature.to(args.device).float()
    adj = data.adj.to(args.device).float()
    label = data.label

    with torch.no_grad():
        _, _, _, _, z = model.ae(feature)

    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    acc_max, embedding = 0, 0
    acc_max_corresponding_metrics = []
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        x_bar, q, pred, _, embedding = model(feature, adj)
        p = data_processor.target_distribution(q.data)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, feature)
        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            _, _, pred, _, embedding = model(feature, adj)
            y_pred = pred.data.cpu().numpy().argmax(1)  # Z
            acc, nmi, ari, f1 = eva(label, y_pred)
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
