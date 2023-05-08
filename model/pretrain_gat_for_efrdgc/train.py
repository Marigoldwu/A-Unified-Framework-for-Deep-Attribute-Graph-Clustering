# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 10:09 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from module.GAT_for_EFRDGC import GAT
from torch.optim import Adam
from utils import data_processor
from utils.evaluation import eva
from utils.utils import get_format_variables


def train(args, feature, label, adj, logger):
    args.hidden_1_dim = 1024
    args.hidden_2_dim = 256
    args.hidden_3_dim = 16
    args.alpha = 0.2
    args.weight_decay = 5e-3
    args.pretrain_epoch = 50
    args.pretrain_lr = 1e-3

    pretrain_gae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = GAT(args.input_dim, args.hidden_1_dim, args.hidden_2_dim, args.hidden_3_dim, args.alpha).to(args.device)
    logger.info(model)
    optimizer = Adam(model.parameters(), args.pretrain_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pretrain_epoch)

    M = data_processor.get_M(adj, args.t).to(args.device)

    adj = data_processor.numpy_to_torch(adj).to(args.device).float()
    adj_label = adj

    data = data_processor.numpy_to_torch(feature).to(args.device).float()

    acc_max = 0
    acc_max_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.pretrain_epoch + 1):
        model.train()
        A_pred, _ = model(data, adj, M, 0, 0, 0, 0)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            A_pred, r = model(data, adj, M, 0, 0, 0, 0)
            kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(r.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(label, kmeans.labels_)
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    torch.save(model.state_dict(), pretrain_gae_filename)
    return r, acc_max_corresponding_metrics
