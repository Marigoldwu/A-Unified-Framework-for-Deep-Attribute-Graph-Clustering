# -*- coding: utf-8 -*-
"""
@Time: 2023/6/28 19:07 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
from model.AGCDRR.model import IGAE, IGAE_encoder, ViewLearner
from sklearn.decomposition import PCA

from utils.data_processor import normalize_adj_torch, normalize_adj, numpy_to_torch, torch_to_numpy
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables, count_parameters


def new_graph(edge_index, weight, n, device):
    edge_index = edge_index.cpu().numpy()
    indices = torch.from_numpy(
        np.vstack((edge_index[0], edge_index[1])).astype(np.int64)).to(device)
    values = weight
    shape = torch.Size((n, n))
    return torch.sparse.FloatTensor(indices, values, shape)


def get_edge_index(adj):
    # Convert to sparse matrix
    sparse_matrix = sp.coo_matrix(torch_to_numpy(adj.cpu()))
    return [sparse_matrix.row, sparse_matrix.col]


def train(args, data, logger):
    args.n_components = 50
    args.gae_n_enc_1 = 1000
    args.gae_n_enc_2 = 500
    args.gae_n_enc_3 = 500
    args.embedding_dim = 500
    args.lr = 5e-4
    args.view_lr = 1e-4
    args.max_epoch = 400
    args.reg_lambda = 1
    pca1 = PCA(n_components=args.n_components)
    x1 = pca1.fit_transform(data.feature)
    x1 = numpy_to_torch(x1).to(args.device).float()

    adj = data.adj.to(args.device).float()
    adj_norm = normalize_adj_torch(adj)
    edge_index = get_edge_index(adj.fill_diagonal_(0))

    model = IGAE(
        gae_n_enc_1=args.gae_n_enc_1,
        gae_n_enc_2=args.gae_n_enc_2,
        gae_n_enc_3=args.gae_n_enc_3,
        n_input=args.n_components,
        clusters=args.clusters
    ).to(args.device)

    view_learner = ViewLearner(
        IGAE_encoder(gae_n_enc_1=args.gae_n_enc_1,
                     gae_n_enc_2=args.gae_n_enc_2,
                     gae_n_enc_3=args.gae_n_enc_3,
                     n_input=args.n_components),
        embedding_dim=args.embedding_dim
    ).to(args.device)

    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)
    optimizer = Adam(model.parameters(), lr=args.lr)

    acc_max, max_embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch + 1):
        view_learner.train()
        view_learner.zero_grad()
        model.eval()
        z_igae, c = model(x1, adj_norm)
        n = z_igae.shape[0]
        edge_logits = view_learner(x1, adj_norm, edge_index)
        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

        aug_adj = new_graph(numpy_to_torch(np.array(edge_index)).to('cuda'), batch_aug_edge_weight, n, 'cuda')
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * adj
        aug_adj = aug_adj.cpu().detach().numpy() + np.eye(n)
        aug_adj = torch.from_numpy(normalize_adj(aug_adj)).to(torch.float32).to('cuda')

        aug_z_igae, aug_c = model(x1, aug_adj)

        edge_drop_out_prob = 1 - batch_aug_edge_weight
        reg = edge_drop_out_prob.mean()

        view_loss = -1 * (args.reg_lambda * reg) + model.calc_loss(c.T, aug_c.T) + model.calc_loss(c, aug_c)

        view_loss.backward()
        view_optimizer.step()

        view_learner.eval()

        model.train()
        model.zero_grad()
        z_igae, c = model(x1, adj_norm)

        n = z_igae.shape[0]
        # with torch.no_grad():
        edge_logits = view_learner(x1, adj_norm, edge_index)

        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

        aug_adj = new_graph(numpy_to_torch(np.array(edge_index)).to('cuda'), batch_aug_edge_weight, n, 'cuda')
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * adj
        aug_adj = aug_adj.cpu().detach().numpy() + np.eye(n)
        aug_adj = torch.from_numpy(normalize_adj(aug_adj)).to(torch.float32).to('cuda')

        aug_z_igae, aug_c = model(x1, aug_adj)

        z_mat = torch.matmul(z_igae, aug_z_igae.T)

        model_loss = model.calc_loss(c.T, aug_c.T) + F.mse_loss(z_mat, torch.eye(n).to('cuda')) + model.calc_loss(c,
                                                                                                                  aug_c)
        model_loss.backward()
        optimizer.step()
        model.eval()

        z = (c + aug_c) / 2
        i = z.argmax(dim=-1)
        acc, nmi, ari, f1 = eva(data.label, i.data.cpu().numpy())
        if acc > acc_max:
            acc_max = acc
            max_acc_corresponding_metrics = [acc, nmi, ari, f1]
            max_embedding = z_mat
        logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                         ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    # Sort F based on the sort indices
    sort_indices = np.argsort(data.label)
    max_embedding = max_embedding[sort_indices]
    result = Result(embedding=max_embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str((count_parameters(model)) +
                                                            count_parameters(view_learner)) + "M(1e6).")
    mem_used = torch.cuda.memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The total memory allocated to model is: {mem_used:.2f} MB.")
    return result
