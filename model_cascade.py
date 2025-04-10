#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: yanms
# @Date  : 2021/11/1 16:16
# @Desc  : CRGCN
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss
from detail.contrast import Contrast


class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):

        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            # x = self.dropout(x)
        return x


class CRGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(CRGCN, self).__init__()

        self.device = args.device
        self.layers = args.layers

        self.tua0 = args.tua0
        self.tua1 = args.tua1
        self.tua2 = args.tua2
        self.tua3 = args.tua3
        self.tua4 = args.tua4
        self.tua5 = args.tua5
        self.lamda = args.lamda
        self.pool = args.pool

        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior in enumerate(self.behaviors)
        })


        self.reg_weight = args.reg_weight
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_all_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)


    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)


    def gcn_propagate(self):
        """
        gcn propagate in each behavior
        """
        all_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for behavior in self.behaviors:
            layer_embeddings = total_embeddings
            indices = self.edge_index[behavior].to(self.device)
            layer_embeddings = self.Graph_encoder[behavior](layer_embeddings, indices)
            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            total_embeddings = layer_embeddings + total_embeddings
            all_embeddings[behavior] = total_embeddings
        return all_embeddings

    def forward(self, batch_data):

        self.storage_all_embeddings = None
        all_embeddings = self.gcn_propagate()
        total_loss = 0
        info_NCE = 0
        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])

            user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
            item_feature = item_all_embedding[items]
            # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)
            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])

        u_p, i_p = torch.split(all_embeddings['buy'], [self.n_users + 1, self.n_items + 1])
        users_feature_p = u_p[users.view(-1)]
        item_feature_p = []
        for i in range(items.shape[1]):
            item_feature_p.append(i_p[items[:, i].view(-1)])

        u_c, i_c = torch.split(all_embeddings['cart'], [self.n_users + 1, self.n_items + 1])
        users_feature_c = u_c[users.view(-1)]
        item_feature_c = []
        for i in range(items.shape[1]):
            item_feature_c.append(i_c[items[:, i].view(-1)])

        u_v, i_v = torch.split(all_embeddings['view'], [self.n_users + 1, self.n_items + 1])
        users_feature_v = u_v[users.view(-1)]
        item_feature_v = []
        for i in range(items.shape[1]):
            item_feature_v.append(i_v[items[:, i].view(-1)])

        u_co, i_co = torch.split(all_embeddings['collect'], [self.n_users + 1, self.n_items + 1])
        users_feature_co = u_co[users.view(-1)]
        item_feature_co = []
        for i in range(items.shape[1]):
            item_feature_co.append(i_co[items[:, i].view(-1)])

        batch_size = len(users)
        adj_u = torch.eye(batch_size).to(self.device)
        adj_i = torch.eye(batch_size).to(self.device)

        contr0 = Contrast(self.tua0)
        contr1 = Contrast(self.tua1)
        contr2 = Contrast(self.tua2)
        contr3 = Contrast(self.tua3)
        contr4 = Contrast(self.tua4)
        contr5 = Contrast(self.tua5)

        l0 = contr0.forward(users_feature_p, users_feature_c, adj_u)
        l1 = contr1.forward(users_feature_p, users_feature_v, adj_u)
        l2 = contr2.forward(item_feature_p[0], item_feature_c[0], adj_i)
        l3 = contr3.forward(item_feature_p[0], item_feature_v[0], adj_i)
        l4 = contr4.forward(users_feature_p, users_feature_co, adj_u)
        l5 = contr5.forward(item_feature_p[0], item_feature_co[0], adj_i)
        info_NCE = l0 + l1 + l2 + l3 + l4 + l5

        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight) + self.lamda * info_NCE

        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]], [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))
        return scores

