# -*- coding: utf-8 -*-

import torch

from torch import nn, Tensor
import torch.nn.functional as F

from ctp.geometric.gat import EdgeGATConv
from ctp.geometric.gcn import EdgeGCNConv


class GATEncoder(nn.Module):
    def __init__(self,
                 nb_nodes: int,
                 nb_edge_types: int,
                 embedding_dim: int = 100,
                 edge_dim: int = 20,
                 nb_heads: int = 3,
                 dropout: float = 0.0,
                 nb_message_rounds: int = 3):
        super().__init__()
        self.nb_message_rounds = nb_message_rounds
        self.embedding = torch.nn.Embedding(num_embeddings=nb_nodes, embedding_dim=embedding_dim, max_norm=1)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.edge_embedding = torch.nn.Embedding(nb_edge_types, edge_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight)
        self.att1 = EdgeGATConv(embedding_dim, embedding_dim, edge_dim, heads=nb_heads, dropout=dropout)
        self.att2 = EdgeGATConv(embedding_dim, embedding_dim, edge_dim)

    def forward(self, batch, slices) -> Tensor:
        x = self.embedding(batch.x).squeeze(1)
        edge_attr = self.edge_embedding(batch.edge_attr).squeeze(1)
        for nr in range(self.nb_message_rounds):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.att1(x, batch.edge_index, edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.att2(x, batch.edge_index, edge_attr)
        chunks = torch.split(x, slices, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        x = torch.cat(chunks, dim=0)
        return x


class GCNEncoder(nn.Module):
    def __init__(self,
                 nb_nodes: int,
                 nb_edge_types: int,
                 embedding_dim: int = 100,
                 edge_dim: int = 20,
                 nb_message_rounds: int = 3):
        super().__init__()
        self.nb_message_rounds = nb_message_rounds
        self.embedding = torch.nn.Embedding(num_embeddings=nb_nodes, embedding_dim=embedding_dim, max_norm=1)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.edge_embedding = torch.nn.Embedding(nb_edge_types, edge_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight)
        self.att1 = EdgeGCNConv(embedding_dim, embedding_dim, edge_dim)
        self.att2 = EdgeGCNConv(embedding_dim, embedding_dim, edge_dim)

    def forward(self, batch, slices) -> Tensor:
        x = self.embedding(batch.x).squeeze(1)
        edge_attr = self.edge_embedding(batch.edge_attr).squeeze(1)
        for nr in range(self.nb_message_rounds):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.att1(x, batch.edge_index, edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.att2(x, batch.edge_index, edge_attr)
        chunks = torch.split(x, slices, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        x = torch.cat(chunks, dim=0)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 target_size: int = 20):
        super().__init__()
        self.target_size = target_size
        self.linear = None

    def query(self,
              graph_emb: Tensor,
              query_edge: Tensor) -> Tensor:
        query = query_edge.squeeze(1).unsqueeze(2).repeat(1, 1, graph_emb.size(2))
        query_emb = torch.gather(graph_emb, 1, query)
        res = query_emb.view(graph_emb.size(0), -1)
        return res

    def forward(self,
                graph_emb: Tensor,
                query_emb: Tensor):
        node_avg = torch.mean(graph_emb, 1)
        node_cat = torch.cat([node_avg, query_emb], -1)
        if self.linear is None:
            self.linear = nn.Linear(node_cat.shape[1], self.target_size)
        res = self.linear(node_cat)
        return res
