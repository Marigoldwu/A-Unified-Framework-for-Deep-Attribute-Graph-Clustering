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

from torch.optim import Adam
from model.DAEGC.model import DAEGC
from utils import data_processor, formatter
from sklearn.cluster import KMeans
from utils.evaluation import eva
from utils.parameter_counter import count_parameters


def train(args, feature, label, adj, logger):
    args.hidden_size = 256
    args.embedding_size = 16
    args.alpha = 0.2
    args.weight_decay = 5e-3
    pretrain_gae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.clusters).to(args.device)
    logger.info(model)
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")

    model.gat.load_state_dict(torch.load(pretrain_gae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    M = data_processor.get_M(adj).to(args.device)

    # data and label
    data = data_processor.numpy_to_torch(feature).to(args.device).float()
    label = label

    adj = data_processor.numpy_to_torch(adj).to(args.device).float()

    adj_label = adj

    with torch.no_grad():
        _, z = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    Q = 0
    acc_max = 0
    acc_max_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(data, adj, M)

            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(label, q)
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(formatter.get_format_variables(epoch="{:0>3d}".format(epoch),
                                                       acc="{:0>.4f}".format(acc),
                                                       nmi="{:0>.4f}".format(nmi),
                                                       ari="{:0>.4f}".format(ari),
                                                       f1="{:0>.4f}".format(f1)))

        A_pred, z, q = model(data, adj, M)
        p = data_processor.target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return Q, acc_max_corresponding_metrics
