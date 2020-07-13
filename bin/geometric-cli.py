#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch_geometric.data import Data as GeometricData, Batch

from kbcr.util import make_batches
from kbcr.clutrr import Data, Instance

from kbcr.geometric import GraphAttentionNetwork
from kbcr.geometric import GraphConvolutionalNetwork

from kbcr.geometric import VecBaselineNetworkV1
from kbcr.geometric import VecBaselineNetworkV2
from kbcr.geometric import Seq2VecEncoderFactory

from typing import Dict, List, Tuple

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=256, precision=4, suppress=True, threshold=sys.maxsize)

torch.set_num_threads(multiprocessing.cpu_count())

# PYTHONPATH=. python3 ./bin/geometric-cli.py
#  --train data/clutrr-emnlp/data_089907f8/*train*
#  --test data/clutrr-emnlp/data_089907f8/*test*


def to_data(instance: Instance,
            relation_to_idx: Dict[str, int],
            nb_entities: int) -> Tuple[GeometricData, Tuple[int, int]]:
    entity_lst = sorted({x for t in instance.story for x in {t[0], t[2]}})
    entity_to_idx = {e: i for i, e in enumerate(entity_lst)}

    x = torch.arange(nb_entities).view(-1, 1)
    edge_list = [(entity_to_idx[s], entity_to_idx[o]) for (s, _, o) in instance.story]
    edge_index = torch.LongTensor(list(zip(*edge_list)))

    edge_types = [relation_to_idx[p] for (_, p, _) in instance.story]
    edge_attr = torch.LongTensor(edge_types).view(-1, 1)

    y = torch.tensor([relation_to_idx[instance.target[1]]])
    res = GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    target_pair = (entity_to_idx[instance.target[0]], entity_to_idx[instance.target[2]])
    return res, target_pair


def to_batches(instances: List[Instance],
               batch_size: int,
               relation_to_idx: Dict[str, int]) -> List[Tuple[Batch, List[int], Tensor, List[Instance]]]:
    nb_instances, res = len(instances), []
    batches = make_batches(nb_instances, batch_size)
    for batch_start, batch_end in batches:
        batch_instances = instances[batch_start:batch_end]
        max_nb_entities = max(i.nb_nodes for i in batch_instances)
        this_batch_size = len(batch_instances)

        batch_pairs = [to_data(i, relation_to_idx, max_nb_entities) for i in batch_instances]
        batch_data: List[GeometricData] = [d for d, _ in batch_pairs]
        batch_targets: List[List[int]] = [[p[0], p[1]] for _, p in batch_pairs]

        max_node = max(i + 1 for b in batch_data for i in b.x[:, 0].cpu().numpy())
        batch = Batch.from_data_list(batch_data)
        slices = [max_node for _ in batch_data]
        targets = torch.LongTensor(batch_targets).view(this_batch_size, 1, 2)

        res += [(batch, slices, targets, batch_instances)]
    return res


def main(argv):
    argparser = argparse.ArgumentParser('Geometric CLUTRR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    train_path = "data/clutrr-emnlp/data_test/64.csv"

    argparser.add_argument('--train', action='store', type=str, default=train_path)
    argparser.add_argument('--test', nargs='+', type=str, default=[])

    argparser.add_argument('--model', '-m', action='store', type=str, default='gat')

    # training params
    argparser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    argparser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.001)
    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=100)

    argparser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    argparser.add_argument('--edge-embedding-size', '-K', action='store', type=int, default=20)
    argparser.add_argument('--hidden-size', action='store', type=int, default=100)
    argparser.add_argument('--nb-filters', action='store', type=int, default=100)
    argparser.add_argument('--nb-heads', action='store', type=int, default=3)
    argparser.add_argument('--nb-rounds', action='store', type=int, default=3)
    argparser.add_argument('--nb-highway', action='store', type=int, default=2)

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--v2', action='store_true', default=False)

    args = argparser.parse_args(argv)

    train_path = args.train
    test_paths = args.test

    model_name = args.model

    nb_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    embedding_size = args.embedding_size
    edge_embedding_size = args.edge_embedding_size
    hidden_size = args.hidden_size
    nb_filters = args.nb_filters
    nb_heads = args.nb_heads
    nb_rounds = args.nb_rounds
    nb_highway = args.nb_highway

    seed = args.seed

    is_v2 = args.v2

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    data = Data(train_path=train_path, test_paths=test_paths)

    entity_lst, _, relation_lst = data.entity_lst, data.predicate_lst, data.relation_lst
    relation_to_idx = {r: i for i, r in enumerate(relation_lst)}

    nb_nodes = len(entity_lst)
    nb_edge_types = len(relation_lst)

    nb_instances = len(data.train)
    batches = to_batches(data.train, batch_size=batch_size, relation_to_idx=relation_to_idx)

    if model_name in {'gat'}:
        model = GraphAttentionNetwork(nb_nodes=nb_nodes, nb_edge_types=nb_edge_types, nb_heads=nb_heads,
                                      embedding_size=embedding_size, edge_embedding_size=edge_embedding_size,
                                      nb_rounds=nb_rounds)
    elif model_name in {'gcn'}:
        model = GraphConvolutionalNetwork(nb_nodes=nb_nodes, nb_edge_types=nb_edge_types, embedding_size=embedding_size,
                                          edge_embedding_size=edge_embedding_size, nb_rounds=nb_rounds)
    else:
        encoder_factory = Seq2VecEncoderFactory()
        encoder = encoder_factory.build(name=model_name, embedding_dim=embedding_size, hidden_size=hidden_size,
                                        num_filters=nb_filters, num_heads=nb_heads, num_highway=nb_highway)

        if is_v2 is False:
            model = VecBaselineNetworkV1(nb_nodes=nb_nodes, nb_edge_types=nb_edge_types, relation_lst=relation_lst,
                                         encoder=encoder, embedding_size=embedding_size)
        else:
            model = VecBaselineNetworkV2(nb_nodes=nb_nodes, nb_edge_types=nb_edge_types, relation_lst=relation_lst,
                                         encoder=encoder, embedding_size=embedding_size)

    params_lst = nn.ParameterList([p for p in model.parameters()])
    optimizer = torch.optim.Adam(params_lst, lr=learning_rate)

    def test(test_set) -> float:
        correct = 0
        model.eval()

        test_batches = to_batches(test_set, batch_size=batch_size, relation_to_idx=relation_to_idx)
        for test_batch, test_slices, test_targets, test_instances in test_batches:
            test_logits = model(test_batch, test_slices, test_targets, test_instances)
            test_predictions = test_logits.max(dim=1)[1]
            correct += test_predictions.eq(test_batch.y).sum().item()
        return correct / len(test_set)

    for epoch in range(1, nb_epochs + 1):
        loss_total = 0.0
        model.train()

        for batch, slices, targets, instances in batches:
            logits = model(batch, slices, targets, instances)

            loss = F.cross_entropy(logits, batch.y, reduction='sum')
            loss_total += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_accuracy = test(data.train)
        print(f'Epoch: {epoch:03d}, Train Loss: {loss_total / nb_instances:.7f}, Accuracy: {train_accuracy:.7f}')

        for name in data.test:
            test_accuracy = test(data.test[name])
            print(f'Epoch: {epoch:03d}, Test Set: {name}, Accuracy: {test_accuracy:.7f}')

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
