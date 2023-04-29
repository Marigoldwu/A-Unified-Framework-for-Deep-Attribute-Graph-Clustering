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
from utils import data_processor, formatter
from utils.evaluation import eva
from utils.parameter_counter import count_parameters


def train(args, feature, label, adj, logger):
    args.embedding_dim = 10
    args.enc_1_dim = 500
    args.enc_2_dim = 500
    args.enc_3_dim = 2000
    args.dec_1_dim = 2000
    args.dec_2_dim = 500
    args.dec_3_dim = 500

    model = SDCN(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                 enc_1_dim=args.enc_1_dim, enc_2_dim=args.enc_2_dim, enc_3_dim=args.enc_3_dim,
                 dec_1_dim=args.dec_1_dim, dec_2_dim=args.dec_2_dim, dec_3_dim=args.dec_3_dim,
                 clusters=args.clusters,
                 v=1).to(args.device)
    pretrain_ae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"

    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")

    model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr)

    data = data_processor.numpy_to_torch(feature).to(args.device).float()
    adj = data_processor.normalize_adj(adj)
    adj = data_processor.numpy_to_torch(adj).to(args.device).float()
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    p, tmp_q = 0, 0
    acc_max = 0
    acc_max_corresponding_metrics = []
    for epoch in range(1, args.max_epoch + 1):
        if epoch % args.update_interval == 0:
            # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data

            p = data_processor.target_distribution(tmp_q)
            y_pred = pred.data.cpu().numpy().argmax(1)  # Z
            acc, nmi, ari, f1 = eva(label, y_pred)
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(formatter.get_format_variables(epoch="{:0>3d}".format(epoch),
                                                       acc="{:0>.4f}".format(acc),
                                                       nmi="{:0>.4f}".format(nmi),
                                                       ari="{:0>.4f}".format(ari),
                                                       f1="{:0>.4f}".format(f1)))

        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tmp_q, acc_max_corresponding_metrics
