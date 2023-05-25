# -*- coding: utf-8 -*-
"""
@Time: 2023/5/8 16:51 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from module.GAE import GAE
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables


def train(args, data, logger):
    args.hidden_dim = 256
    args.embedding_dim = 16
    args.weight_decay = 5e-3
    args.adj_loop = True
    args.adj_norm = True
    args.adj_symmetric = True
    pretrain_gae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = GAE(args.input_dim, args.hidden_dim, args.embedding_dim).to(args.device)
    logger.info(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    feature = data.feature.to(args.device).float()
    adj = data.adj.to(args.device).float()
    adj_label = adj
    label = data.label

    acc_max, embedding = 0, None
    acc_max_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.pretrain_epoch+1):
        model.train()
        A_pred, embedding = model(feature, adj)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            _, embedding = model(feature, adj)
            kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(embedding.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(label, kmeans.labels_)
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    torch.save(model.state_dict(), pretrain_gae_filename)
    result = Result(embedding=embedding, acc_max_corresponding_metrics=acc_max_corresponding_metrics)
    return result
