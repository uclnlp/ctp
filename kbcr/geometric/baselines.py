# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from torch_geometric.data import Batch

from kbcr.util import pad_sequences
from kbcr.clutrr import Instance
from kbcr.geometric.models import GATEncoder
from kbcr.geometric.models import GCNEncoder
from kbcr.geometric.models import Decoder

from typing import List, Callable, Any


class GraphAttentionNetwork(nn.Module):
    def __init__(self,
                 nb_nodes: int,
                 nb_edge_types: int,
                 nb_heads: int = 3,
                 embedding_size: int = 100,
                 edge_embedding_size: int = 20,
                 nb_rounds: int = 3):
        super().__init__()
        self.encoder = GATEncoder(nb_nodes=nb_nodes, nb_edge_types=nb_edge_types, nb_heads=nb_heads,
                                  embedding_dim=embedding_size, edge_dim=edge_embedding_size,
                                  nb_message_rounds=nb_rounds)
        self.decoder = Decoder(target_size=nb_edge_types)

    def forward(self,
                batch: Batch,
                slices: List[int],
                targets: Tensor,
                instances: List[Instance]) -> Tensor:
        graph_emb = self.encoder(batch, slices)
        query_emb = self.decoder.query(graph_emb, targets)
        logits = self.decoder(graph_emb, query_emb)
        return logits


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self,
                 nb_nodes: int,
                 nb_edge_types: int,
                 embedding_size: int = 100,
                 edge_embedding_size: int = 20,
                 nb_rounds: int = 3):
        super().__init__()
        self.encoder = GCNEncoder(nb_nodes=nb_nodes, nb_edge_types=nb_edge_types, embedding_dim=embedding_size,
                                  edge_dim=edge_embedding_size, nb_message_rounds=nb_rounds)
        self.decoder = Decoder(target_size=nb_edge_types)

    def forward(self,
                batch: Batch,
                slices: List[int],
                targets: Tensor,
                instances: List[Instance]) -> Tensor:
        graph_emb = self.encoder(batch, slices)
        query_emb = self.decoder.query(graph_emb, targets)
        logits = self.decoder(graph_emb, query_emb)
        return logits


class VecBaselineNetworkV1(nn.Module):
    ENTITY_PREFIX = "ENTITY_"
    UNK = 'UNK'

    def __init__(self,
                 nb_nodes: int,
                 nb_edge_types: int,
                 relation_lst: List[str],
                 encoder: Callable[[Tensor, Any], Tensor],
                 embedding_size: int = 100):
        super().__init__()
        self.nb_nodes = nb_nodes
        self.nb_edge_types = nb_edge_types
        self.relation_lst = relation_lst

        self.entity_lst = [f'{self.ENTITY_PREFIX}{i}' for i in range(self.nb_nodes)]
        self.symbol_lst = sorted({s for s in self.entity_lst + self.relation_lst} | {'UNK'})

        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbol_lst)}

        nb_symbols = len(self.symbol_lst)
        self.symbol_embeddings = nn.Embedding(nb_symbols, embedding_size, sparse=False)

        self.encoder = encoder
        self.projection = None

    def normalise(self, instance: Instance):
        entity_lst = sorted({e for t in instance.story + [instance.target] for e in {t[0], t[2]}})
        entity_to_symbol = {e: f'{self.ENTITY_PREFIX}{i}' for i, e in enumerate(entity_lst)}
        new_story = [(entity_to_symbol[t[0]], t[1], entity_to_symbol[t[2]]) for t in instance.story]
        new_target = (entity_to_symbol[instance.target[0]], instance.target[1], entity_to_symbol[instance.target[2]])
        return Instance(new_story, new_target, instance.nb_nodes)

    def forward(self,
                batch: Batch,
                slices: List[int],
                targets: Tensor,
                instances: List[Instance]) -> Tensor:
        linear_story_lst = []
        target_lst = []

        instances = [self.normalise(instance) for instance in instances]
        for instance in instances:
            linear_story = [self.symbol_to_idx[s] for t in instance.story for s in t]
            linear_story_lst += [linear_story]

            target = [
                self.symbol_to_idx[instance.target[0]],
                self.symbol_to_idx[instance.target[2]]
            ]
            target_lst += [target]

        story_padded = pad_sequences(linear_story_lst, value=self.symbol_to_idx['UNK'])
        batch_linear_story = torch.LongTensor(story_padded)
        batch_target = torch.LongTensor(target_lst)

        batch_linear_story_emb = self.symbol_embeddings(batch_linear_story)
        batch_target_emb = self.symbol_embeddings(batch_target)

        story_code = self.encoder(batch_linear_story_emb, None)
        target_code = self.encoder(batch_target_emb, None)

        if self.projection is None:
            in_dim = story_code.shape[-1] + target_code.shape[-1]
            self.projection = nn.Linear(in_dim, self.nb_edge_types)

        story_target_code = torch.cat([story_code, target_code], dim=-1)
        logits = self.projection(story_target_code)
        return logits


class VecBaselineNetworkV2(nn.Module):
    ENTITY_PREFIX = "ENTITY_"
    UNK = 'UNK'

    def __init__(self,
                 nb_nodes: int,
                 nb_edge_types: int,
                 relation_lst: List[str],
                 encoder: Callable[[Tensor, Any], Tensor],
                 embedding_size: int = 100):
        super().__init__()
        self.nb_nodes = nb_nodes
        self.nb_edge_types = nb_edge_types
        self.relation_lst = relation_lst

        self.entity_lst = [f'{self.ENTITY_PREFIX}{i}' for i in range(self.nb_nodes)]
        self.symbol_lst = sorted({s for s in self.entity_lst + self.relation_lst} | {'UNK'})

        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbol_lst)}

        nb_symbols = len(self.symbol_lst)
        self.symbol_embeddings = nn.Embedding(nb_symbols, embedding_size, sparse=False)

        self.encoder = encoder
        self.projection = None

    def normalise(self, instance: Instance):
        entity_lst = sorted({e for t in instance.story + [instance.target] for e in {t[0], t[2]}})
        entity_to_symbol = {e: f'{self.ENTITY_PREFIX}{i}' for i, e in enumerate(entity_lst)}
        new_story = [(entity_to_symbol[t[0]], t[1], entity_to_symbol[t[2]]) for t in instance.story]
        new_target = (entity_to_symbol[instance.target[0]], instance.target[1], entity_to_symbol[instance.target[2]])
        return Instance(new_story, new_target, instance.nb_nodes)

    def forward(self,
                batch: Batch,
                slices: List[int],
                targets: Tensor,
                instances: List[Instance]) -> Tensor:
        linear_story_lst = []

        instances = [self.normalise(instance) for instance in instances]
        for instance in instances:
            seq = [instance.target] + instance.story
            linear_story = [self.symbol_to_idx[s] for t in seq for s in t]
            linear_story_lst += [linear_story]

        seq_padded = pad_sequences(linear_story_lst, value=self.symbol_to_idx['UNK'])
        batch_seq = torch.LongTensor(seq_padded)
        batch_seq_emb = self.symbol_embeddings(batch_seq)
        seq_code = self.encoder(batch_seq_emb, None)

        if self.projection is None:
            in_dim = seq_code.shape[-1]
            self.projection = nn.Linear(in_dim, self.nb_edge_types)

        logits = self.projection(seq_code)
        return logits
