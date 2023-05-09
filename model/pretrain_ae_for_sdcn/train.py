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
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from utils.evaluation import eva
from module.AE import AE
from utils import load_data, data_processor
from utils.utils import get_format_variables


def train(args, data, logger):
    """
    Using this function can pretrain the AE module and save the parameters file to specify directory

    :param args: the settings of the input model
    :param data:
    :param logger: the logger object created in main.py
    """
    args.embedding_dim = 10
    args.enc_1_dim = 500
    args.enc_2_dim = 500
    args.enc_3_dim = 2000
    args.dec_1_dim = 2000
    args.dec_2_dim = 500
    args.dec_3_dim = 500
    pretrain_ae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    logger.info("The pretrain .pkl file will be saved to the path: " + pretrain_ae_filename)
    model = AE(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
               enc_1_dim=args.enc_1_dim, enc_2_dim=args.enc_2_dim, enc_3_dim=args.enc_3_dim,
               dec_1_dim=args.dec_1_dim, dec_2_dim=args.dec_2_dim, dec_3_dim=args.dec_3_dim).to(args.device)

    dataset = load_data.LoadDataset(data.feature)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    logger.info(model)
    optimizer = Adam(model.parameters(), args.pretrain_lr)
    acc_max = 0
    acc_max_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.pretrain_epoch + 1):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(args.device)
            x_bar, _, _, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = data_processor.numpy_to_torch(data.feature).to(args.device).float()
            x_bar, _, _, _, z = model(x)
            kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(z.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(data.label, kmeans.labels_)
            if acc > acc_max:
                acc_max = acc
                acc_max_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    torch.save(model.state_dict(), pretrain_ae_filename)
    return z, acc_max_corresponding_metrics
