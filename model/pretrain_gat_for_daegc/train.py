# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 9:46 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from module.GAT_for_DAEGC import GAT
from torch.optim import Adam
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables


def train(args, data, logger):
    args.hidden_size = 256
    args.embedding_size = 16
    args.alpha = 0.2
    args.weight_decay = 5e-3

    pretrain_gat_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = GAT(num_features=args.input_dim,
                hidden_size=args.hidden_size,
                embedding_size=args.embedding_size,
                alpha=args.alpha).to(args.device)
    logger.info(model)
    optimizer = Adam(model.parameters(), args.pretrain_lr)

    M = data.M.to(args.device)

    adj = data.adj.to(args.device).float()
    adj_label = adj

    feature = data.feature.to(args.device).float()
    label = data.label

    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.pretrain_epoch + 1):
        model.train()
        A_pred, embedding = model(feature, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
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

    torch.save(model.state_dict(), pretrain_gat_filename)
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    return result
