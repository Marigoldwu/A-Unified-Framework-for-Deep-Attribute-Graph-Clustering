# -*- coding: utf-8 -*-
"""
@Time: 2023/5/20 15:02 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from module.AE_for_DFCN import AE
from torch.optim import Adam
from sklearn.cluster import KMeans
from utils import load_data
from utils.data_processor import numpy_to_torch
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables


def train(args, data, logger):
    param_dict = {'acm': [30, 5e-5, 100],
                  'cite': [30, 1e-4, 100],
                  'cora': [30, 1e-4, 50],
                  'dblp': [30, 1e-4, 50],
                  'reut': [30, 1e-4, 100],
                  'hhar': [30, 1e-3, 50],
                  'usps': [30, 1e-3, 30]}
    args.pretrain_epoch = param_dict[args.dataset_name][0]
    args.pretrain_lr = param_dict[args.dataset_name][1]
    args.n_input = param_dict[args.dataset_name][2]
    args.embedding_dim = 20
    args.ae_n_enc_1 = 128
    args.ae_n_enc_2 = 256
    args.ae_n_enc_3 = 512
    args.ae_n_dec_1 = 512
    args.ae_n_dec_2 = 256
    args.ae_n_dec_3 = 128

    pretrain_ae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    logger.info("The pretrain .pkl file will be saved to the path: " + pretrain_ae_filename)
    model = AE(args.ae_n_enc_1, args.ae_n_enc_2, args.ae_n_enc_3,
               args.ae_n_dec_1, args.ae_n_dec_2, args.ae_n_dec_3, args.n_input, args.embedding_dim).to(args.device)
    pca = PCA(n_components=args.n_input)
    X_pca = pca.fit_transform(data.feature)
    dataset = load_data.LoadDataset(X_pca)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    logger.info(model)
    optimizer = Adam(model.parameters(), args.pretrain_lr)
    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.pretrain_epoch + 1):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(args.device)
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            _, embedding = model(numpy_to_torch(X_pca).to(args.device).float())
            kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(embedding.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(data.label, kmeans.labels_)
            if acc > acc_max:
                acc_max = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    torch.save(model.state_dict(), pretrain_ae_filename)
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    return result
