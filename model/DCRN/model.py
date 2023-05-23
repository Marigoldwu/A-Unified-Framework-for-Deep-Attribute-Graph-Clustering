# -*- coding: utf-8 -*-
"""
@Time: 2023/5/23 18:19 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from module.AE_for_DFCN import AE
from module.IGAE_for_DCRN import IGAE
from module.Readout import Readout


class DCRN(nn.Module):
    def __init__(self, name,
                 ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, n_node=None):
        super(DCRN, self).__init__()

        # Auto Encoder
        self.ae = AE(ae_n_enc_1=ae_n_enc_1,
                     ae_n_enc_2=ae_n_enc_2,
                     ae_n_enc_3=ae_n_enc_3,
                     ae_n_dec_1=ae_n_dec_1,
                     ae_n_dec_2=ae_n_dec_2,
                     ae_n_dec_3=ae_n_dec_3,
                     n_input=n_input,
                     n_z=n_z)

        # Improved Graph Auto Encoder From DFCN
        self.igae = IGAE(name=name,
                         gae_n_enc_1=gae_n_enc_1,
                         gae_n_enc_2=gae_n_enc_2,
                         gae_n_enc_3=gae_n_enc_3,
                         gae_n_dec_1=gae_n_dec_1,
                         gae_n_dec_2=gae_n_dec_2,
                         gae_n_dec_3=gae_n_dec_3,
                         n_input=n_input)

        # fusion parameter from DFCN
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.zeros(1))

        # cluster layer (clustering assignment matrix)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)

        # readout function
        self.R = Readout(K=n_clusters)

    # calculate the soft assignment distribution Q
    def q_distribute(self, Z, Z_ae, Z_igae):
        """
        calculate the soft assignment distribution based on the embedding and the cluster centers
        Args:
            Z: fusion node embedding
            Z_ae: node embedding encoded by AE
            Z_igae: node embedding encoded by IGAE
        Returns:
            the soft assignment distribution Q
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_layer, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_layer, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]

    def forward(self, X_tilde1, Am, X_tilde2, Ad):
        # node embedding encoded by AE
        Z_ae1 = self.ae.encoder(X_tilde1)
        Z_ae2 = self.ae.encoder(X_tilde2)

        # node embedding encoded by IGAE
        Z_igae1, A_igae1, AZ_1, Z_1 = self.igae.encoder(X_tilde1, Am)
        Z_igae2, A_igae2, AZ_2, Z_2 = self.igae.encoder(X_tilde2, Ad)

        # cluster-level embedding calculated by readout function
        Z_tilde_ae1 = self.R(Z_ae1)
        Z_tilde_ae2 = self.R(Z_ae2)
        Z_tilde_igae1 = self.R(Z_igae1)
        Z_tilde_igae2 = self.R(Z_igae2)

        # linear combination of view 1 and view 2
        Z_ae = (Z_ae1 + Z_ae2) / 2
        Z_igae = (Z_igae1 + Z_igae2) / 2

        # node embedding fusion from DFCN
        Z_i = self.a * Z_ae + self.b * Z_igae
        Z_l = torch.spmm(Am, Z_i)
        S = torch.mm(Z_l, Z_l.t())
        S = F.softmax(S, dim=1)
        Z_g = torch.mm(S, Z_l)
        Z = self.alpha * Z_g + Z_l

        # AE decoding
        X_hat = self.ae.decoder(Z)

        # IGAE decoding
        Z_hat, Z_adj_hat, AZ_de, Z_de = self.igae.decoder(Z, Am)
        sim = (A_igae1 + A_igae2) / 2
        A_hat = sim + Z_adj_hat

        # node embedding and cluster-level embedding
        Z_ae_all = [Z_ae1, Z_ae2, Z_tilde_ae1, Z_tilde_ae2]
        Z_gae_all = [Z_igae1, Z_igae2, Z_tilde_igae1, Z_tilde_igae2]

        # the soft assignment distribution Q
        Q = self.q_distribute(Z, Z_ae, Z_igae)

        # propagated embedding AZ_all and embedding Z_all
        AZ_en = []
        Z_en = []
        for i in range(len(AZ_1)):
            AZ_en.append((AZ_1[i] + AZ_2[i]) / 2)
            Z_en.append((Z_1[i] + Z_2[i]) / 2)
        AZ_all = [AZ_en, AZ_de]
        Z_all = [Z_en, Z_de]

        return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all
