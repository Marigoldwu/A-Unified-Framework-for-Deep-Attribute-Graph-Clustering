# -*- coding: utf-8 -*-
"""
@Time: 2023/6/28 14:25 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from augmentation.fuzzy_graph import similarity
from model.CCGC.model import Encoder_Net
from utils.evaluation import eva
from utils.kmeans_gpu import kmeans
from utils.result import Result
from utils.utils import get_format_variables, count_parameters


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    # reg = [2 / 3] * (layer)
    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs


def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def clustering(feature, true_labels, cluster_num):
    predict_labels, dis, initial = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy())
    return acc, nmi, ari, f1, predict_labels.numpy(), dis


def train(args, data, logger):
    args.t = 4  # number of GNN layers
    args.dims = 500
    if args.input_dim < 500:
        args.dim = args.input_dim
    args.lr = 1e-4
    if args.dataset_name == "dblp":
        args.lr = 1e-3
    args.max_epoch = 400 
    args.threshold = 0.5
    args.alpha = 0.5
    adj = data.adj
    true_labels = data.label
    features = data.feature

    adj_norm_s = preprocess_graph(adj, args.t, norm='sym', renorm=True)
    smooth_fea = sp.csr_matrix(data.feature).toarray()
    for a in adj_norm_s:
        smooth_fea = a.dot(smooth_fea)
    smooth_fea = torch.FloatTensor(smooth_fea)

    # init
    best_acc, best_nmi, best_ari, best_f1, predict_labels, dis = clustering(smooth_fea, true_labels, args.clusters)

    # MLP
    model = Encoder_Net([features.shape[1]] + [args.dims])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # GPU
    model.to(args.device)
    smooth_fea = smooth_fea.to(args.device)
    target = torch.eye(smooth_fea.shape[0]).to(args.device)
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    max_embedding = None
    for epoch in range(args.max_epoch):
        model.train()
        z1, z2 = model(smooth_fea)
        if epoch > 50:
            high_confidence = torch.min(dis, dim=1).values
            threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]

            # pos samples
            index = torch.tensor(range(smooth_fea.shape[0]), device=args.device)[high_confidence_idx]
            y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]
            index = index[torch.argsort(y_sam)]
            class_num = {}

            for label in torch.sort(y_sam).values:
                label = label.item()
                if label in class_num.keys():
                    class_num[label] += 1
                else:
                    class_num[label] = 1
            key = sorted(class_num.keys())
            if len(class_num) < 2:
                continue
            pos_contrastive = 0
            centers_1 = torch.tensor([], device=args.device)
            centers_2 = torch.tensor([], device=args.device)

            for i in range(len(key[:-1])):
                class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
                now = index[class_num[key[i]]:class_num[key[i + 1]]]
                pos_embed_1 = z1[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                pos_embed_2 = z2[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()
                centers_1 = torch.cat([centers_1, torch.mean(z1[now], dim=0).unsqueeze(0)], dim=0)
                centers_2 = torch.cat([centers_2, torch.mean(z2[now], dim=0).unsqueeze(0)], dim=0)

            pos_contrastive = pos_contrastive / args.clusters
            if pos_contrastive == 0:
                continue
            if len(class_num) < 2:
                loss = pos_contrastive
            else:
                centers_1 = F.normalize(centers_1, dim=1, p=2)
                centers_2 = F.normalize(centers_2, dim=1, p=2)
                S = centers_1 @ centers_2.T
                S_diag = torch.diag_embed(torch.diag(S))
                S = S - S_diag
                neg_contrastive = F.mse_loss(S, torch.zeros_like(S))
                loss = pos_contrastive + args.alpha * neg_contrastive

        else:
            S = z1 @ z2.T
            loss = F.mse_loss(S, target)

        loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 1 == 0:
            model.eval()
            z1, z2 = model(smooth_fea)

            hidden_emb = (z1 + z2) / 2

            acc, nmi, ari, f1, predict_labels, dis = clustering(hidden_emb, true_labels, args.clusters)
            if acc >= best_acc:
                best_acc = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
                max_embedding = hidden_emb
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    sort_indices = np.argsort(data.label)
    max_embedding = max_embedding[sort_indices]
    result = Result(embedding=max_embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The total memory allocated to model is: {mem_used:.2f} MB.")
    return result
