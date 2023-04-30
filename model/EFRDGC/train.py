# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 9:36 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F

from model.EFRDGC.model import EFRDGC
from torch.optim import Adam
from sklearn.cluster import KMeans

from utils import data_processor, formatter
from utils.evaluation import eva
from utils.parameter_counter import count_parameters


def train(args, feature, label, adj, logger):
    max_epoch_dict = {"acm": 200,
                      "dblp": 200,
                      "cite": 100,
                      "cora": 50,
                      "usps": 200}
    # configuration of EFRDGC
    args.embedding_dim = 10
    args.enc_1_dim = 1024
    args.enc_2_dim = 256
    args.enc_3_dim = 16
    args.dec_1_dim = 16
    args.dec_2_dim = 256
    args.dec_3_dim = 1024
    args.hidden_1_dim = 1024
    args.hidden_2_dim = 256
    args.hidden_3_dim = 16
    args.sigma = 0.1
    args.alpha = 0.2
    args.lambda1 = 1.0
    args.weight_decay = 5e-3
    args.max_epoch = max_epoch_dict[args.dataset_name]
    # load model
    model = EFRDGC(args.input_dim, args.embedding_dim, args.enc_1_dim, args.enc_2_dim, args.enc_3_dim,
                   args.dec_1_dim, args.dec_2_dim, args.dec_3_dim,
                   args.hidden_1_dim, args.hidden_2_dim, args.hidden_3_dim,
                   args.clusters, alpha=args.alpha).to(args.device)
    logger.info(model)
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    # load pretraining parameters
    pretrain_ae_filename = args.pretrain_ae_save_path + args.dataset_name + ".pkl"
    pretrain_gae_filename = args.pretrain_gae_save_path + args.dataset_name + ".pkl"
    model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))
    model.gae.load_state_dict(torch.load(pretrain_gae_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)

    # data transformation
    M = data_processor.get_M(adj).to(args.device).float()
    feature = data_processor.numpy_to_torch(feature).to(args.device).float()
    adj = data_processor.numpy_to_torch(adj).to(args.device).float()
    adj_label = adj

    # init clustering centers
    with torch.no_grad():
        _, z, _, _ = model(feature, adj, M, args.sigma)
    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    # training
    p, acc_max = 0, 0
    acc_max_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch+1):
        model.train()
        if epoch % args.update_interval == 0:
            A_pred, z, tmp_q, x_bar = model(feature, adj, M, args.sigma)
            tmp_q = tmp_q.data
            p = data_processor.target_distribution(tmp_q)
            y_pred = tmp_q.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, y_pred)
            # record the max value
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(formatter.get_format_variables(epoch="{:0>3d}".format(epoch),
                                                       acc="{:0>.4f}".format(acc),
                                                       nmi="{:0>.4f}".format(nmi),
                                                       ari="{:0>.4f}".format(ari),
                                                       f1="{:0>.4f}".format(f1)))

        A_pred, z, q, x_bar = model(feature, adj, M)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss_gae = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        re_loss_ae = F.mse_loss(x_bar, feature)
        loss = args.lambda1 * kl_loss + 1.0 * re_loss_gae + 1.0 * re_loss_ae

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    mem_used = torch.cuda.memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The total memory allocated to model is: {mem_used:.2f} MB.")
    return z, acc_max_corresponding_metrics
