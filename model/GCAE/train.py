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

from model.GCAE.model import GCAE
from torch.optim import Adam
from sklearn.cluster import KMeans
from utils import data_processor
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables, count_parameters


def train(args, data, logger):
    args.hidden_dim = 256
    args.embedding_dim = 16
    args.weight_decay = 5e-3
    args.max_epoch = 100

    # load model
    model = GCAE(args.input_dim, args.clusters, args.hidden_dim, args.embedding_dim).to(args.device)
    logger.info(model)
    # load pretraining parameters
    pretrain_gae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model.gae.load_state_dict(torch.load(pretrain_gae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data transformation
    feature = data.feature.to(args.device).float()
    adj = data.adj.to(args.device).float()
    label = data.label
    adj_label = adj

    # init clustering centers
    with torch.no_grad():
        _, _, z = model(feature, adj)
    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = []
    for epoch in range(1, args.max_epoch+1):
        model.train()
        A_pred, q, embedding = model(feature, adj)
        p = data_processor.target_distribution(q.data)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            y_pred = q.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, y_pred)
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
    mem_used = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The max memory allocated to model is: {mem_used:.2f} MB.")
    return result
