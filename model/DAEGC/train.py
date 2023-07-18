# -*- coding: utf-8 -*-
"""
@Time: 2023/4/27 17:12 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from model.DAEGC.model import DAEGC
from utils import data_processor
from sklearn.cluster import KMeans
from utils.evaluation import eva
from utils.result import Result
from utils.utils import count_parameters, get_format_variables


def train(args, data, logger):
    args.hidden_size = 256
    args.embedding_size = 16
    args.alpha = 0.2
    args.weight_decay = 5e-3

    pretrain_gae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.clusters).to(args.device)
    logger.info(model)
    model.gat.load_state_dict(torch.load(pretrain_gae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data and label
    feature = data.feature.to(args.device).float()
    M = data.M.to(args.device).float()
    adj = data.adj.to(args.device).float()
    label = data.label

    adj_label = adj

    with torch.no_grad():
        _, z = model.gat(feature, adj, M)

    # using the result of pretraining to init clustering centers
    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        A_pred, embedding, q = model(feature, adj, M)
        p = data_processor.target_distribution(q.data)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred_q = q.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, pred_q)
            if acc > acc_max:
                acc_max = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
                max_embedding = embedding
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    # Sort F based on the sort indices
    sort_indices = np.argsort(data.label)
    max_embedding = max_embedding[sort_indices]
    result = Result(embedding=max_embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The total memory allocated to model is: {mem_used:.2f} MB.")
    return result
