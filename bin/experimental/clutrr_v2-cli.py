#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F

from kbcr.util import make_batches
from kbcr.clutrr import Fact, Data, Instance, accuracy

from kbcr.clutrr.models.smart import NeuralKB
from kbcr.clutrr.models.smart import Hoppy

from kbcr.reformulators import BaseReformulator
from kbcr.reformulators import StaticReformulator
from kbcr.reformulators import LinearReformulator
from kbcr.reformulators import AttentiveReformulator
from kbcr.reformulators import MemoryReformulator

from kbcr.kernels import BaseKernel, GaussianKernel
from kbcr.regularizers import N2, N3, Entropy

from kbcr.visualization import HintonDiagram

from typing import List, Tuple, Dict, Optional

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=256, precision=4, suppress=True, threshold=sys.maxsize)

torch.set_num_threads(multiprocessing.cpu_count())


class AttentiveEmbedding(nn.Module):
    def __init__(self,
                 nb_predicates: int,
                 nb_relations: int,
                 embedding_size: int,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.nb_predicates = nb_predicates
        self.nb_relations = nb_relations
        self.embedding_size = embedding_size

        self.predicate_embeddings = self._param(dim_in=self.nb_predicates, dim_out=self.embedding_size, std=1.0)
        self.predicate_embeddings.requires_grad = False

        self.projection = self._param(dim_in=self.nb_relations, dim_out=self.nb_predicates, std=1.0)
        self.projection.requires_grad = False

        if device is not None:
            self.predicate_embeddings = self.predicate_embeddings.to(device)
            self.projection = self.projection.to(device)

    @staticmethod
    def _param(dim_in: int, dim_out: int, std: float = 1.0):
        w = torch.zeros(dim_in, dim_out)
        nn.init.normal_(w, std=std)
        return nn.Parameter(w)

    @property
    def attention(self) -> Tensor:
        return torch.softmax(self.projection, dim=1)

    @property
    def weight(self) -> Tensor:
        return self.attention @ self.predicate_embeddings

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight, sparse=False)


def decode(vector: Tensor,
           kernel: BaseKernel,
           relation_embeddings: nn.Module) -> Tuple[int, float]:
    weight = relation_embeddings.weight
    k = kernel.pairwise(vector, weight)[0, :]
    top_idx = k.argmax(dim=0).item()
    top_score = k[top_idx].item()
    return top_idx, top_score


def show_rules(model: Hoppy,
               kernel: BaseKernel,
               relation_embeddings: nn.Embedding,
               data: Data,
               relation_to_idx: Dict[str, int],
               device: Optional[torch.device] = None):
    idx_to_relation = {i: r for r, i in relation_to_idx.items()}
    for p in data.predicate_lst:
        r = data.predicate_to_relations[p][0]
        indices = torch.from_numpy(np.array([relation_to_idx[r]], dtype=np.int64))
        if device is not None:
            indices = indices.to(device)
        r_emb = relation_embeddings(indices)
        for hops in model.hops_lst:
            p_hops = []
            for hop in hops:
                hop_vec = hop(r_emb)
                hop_idx, hop_score = decode(hop_vec, kernel, relation_embeddings)
                hop_r = idx_to_relation[hop_idx]
                hop_p = data.relation_to_predicate[hop_r]
                p_hops += [(hop_p, hop_score)]
            print(p, ' ← ', ', '.join(f'({a} {b:.4f})' for a, b in p_hops))
    return


class Batcher:
    def __init__(self,
                 batch_size: int,
                 nb_examples: int,
                 nb_epochs: int,
                 random_state: np.random.RandomState):
        self.batch_size = batch_size
        self.nb_examples = nb_examples
        self.nb_epochs = nb_epochs
        self.random_state = random_state

        size = self.nb_epochs * self.nb_examples
        self.curriculum = np.zeros(size, dtype=np.int32)

        for epoch_no in range(nb_epochs):
            start, end = epoch_no * nb_examples, (epoch_no + 1) * nb_examples
            self.curriculum[start: end] = self.random_state.permutation(nb_examples)

        self.batches = make_batches(self.curriculum.shape[0], self.batch_size)
        self.nb_batches = len(self.batches)

    def get_batch(self,
                  batch_start: int,
                  batch_end: int) -> np.ndarray:
        return self.curriculum[batch_start:batch_end]


def encode_relation(facts: List[Fact],
                    relation_embeddings: nn.Embedding,
                    relation_to_idx: Dict[str, int],
                    device: Optional[torch.device] = None) -> Tensor:
    indices_np = np.array([relation_to_idx[r] for _, r, _ in facts], dtype=np.int64)
    indices = torch.from_numpy(indices_np)
    if device is not None:
        indices = indices.to(device)
    return relation_embeddings(indices)


def encode_arguments(facts: List[Fact],
                     entity_embeddings: nn.Embedding,
                     entity_to_idx: Dict[str, int],
                     device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    indices_np = np.array([[entity_to_idx[s], entity_to_idx[o]] for s, _, o in facts], dtype=np.int64)
    indices = torch.from_numpy(indices_np)
    if device is not None:
        indices = indices.to(device)
    emb = entity_embeddings(indices)
    return emb[:, 0, :], emb[:, 1, :]


def encode_entities(facts: List[Fact],
                    entity_embeddings: nn.Embedding,
                    entity_to_idx: Dict[str, int],
                    device: Optional[torch.device]) -> nn.Embedding:
    indices_lst = sorted({entity_to_idx[e] for s, r, o in facts for e in {s, o}})
    indices = torch.from_numpy(np.array(indices_lst, dtype=np.int64))
    if device is not None:
        indices = indices.to(device)
    emb = nn.Embedding.from_pretrained(embeddings=entity_embeddings(indices), sparse=False, freeze=True)
    return emb


def make_easy(predicate_lst: List[str],
              predicate_to_relations: Dict[str, List[str]],
              relation_to_idx: Dict[str, int],
              relation_embeddings: nn.Module):
    for p in predicate_lst:
        r_lst = predicate_to_relations[p]
        r_idx_lst = [relation_to_idx[r] for r in r_lst]
        for i in r_idx_lst[1:]:
            relation_embeddings.projection.data[i, :] = relation_embeddings.projection.data[r_idx_lst[0], :]
    return


def main(argv):
    argparser = argparse.ArgumentParser('CLUTRR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    train_path = test_path = "data/clutrr-emnlp/data_test/64.csv"

    argparser.add_argument('--train', action='store', type=str, default=train_path)
    argparser.add_argument('--test', nargs='+', type=str, default=[test_path])

    # model params
    argparser.add_argument('--embedding-size', '-k', action='store', type=int, default=20)
    argparser.add_argument('--k-max', '-m', action='store', type=int, default=10)
    argparser.add_argument('--max-depth', '-d', action='store', type=int, default=2)

    argparser.add_argument('--hops', nargs='+', type=str, default=['2', '2', '1R'])

    # training params
    argparser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    argparser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)
    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=8)
    argparser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                           choices=['adagrad', 'adam', 'sgd'])

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--evaluate-every', '-V', action='store', type=int, default=32)

    argparser.add_argument('--N2', action='store', type=float, default=None)
    argparser.add_argument('--N3', action='store', type=float, default=None)
    argparser.add_argument('--entropy', '-E', action='store', type=float, default=None)

    argparser.add_argument('--reformulator', '-r', action='store', type=str, default='linear',
                           choices=['static', 'linear', 'attentive', 'memory'])
    argparser.add_argument('--nb-rules', '-R', action='store', type=int, default=4)

    argparser.add_argument('--slope', '-S', action='store', type=float, default=None)
    argparser.add_argument('--init-size', '-i', action='store', type=float, default=1.0)

    argparser.add_argument('--debug', '-D', action='store_true', default=False)

    args = argparser.parse_args(argv)

    train_path = args.train
    test_paths = args.test

    embedding_size = args.embedding_size

    k_max = args.k_max
    max_depth = args.max_depth

    hops_str = args.hops

    nb_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    optimizer_name = args.optimizer

    seed = args.seed

    evaluate_every = args.evaluate_every

    N2_weight = args.N2
    N3_weight = args.N3
    entropy_weight = args.entropy

    reformulator_name = args.reformulator
    nb_rules = args.nb_rules

    slope = args.slope
    init_size = args.init_size

    is_debug = args.debug

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    data = Data(train_path=train_path, test_paths=test_paths)

    relation_to_predicate = data.relation_to_predicate
    predicate_to_relations = data.predicate_to_relations
    entity_lst, predicate_lst, relation_lst = data.entity_lst, data.predicate_lst, data.relation_lst

    nb_examples = len(data.train)
    nb_entities = len(entity_lst)
    nb_predicates = len(predicate_lst)
    nb_relations = len(relation_lst)

    entity_to_idx = {e: i for i, e in enumerate(entity_lst)}
    relation_to_idx = {r: i for i, r in enumerate(relation_lst)}

    kernel = GaussianKernel(slope=slope)

    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=False).to(device)

    if entropy_weight is None:
        relation_embeddings = nn.Embedding(nb_relations, embedding_size, sparse=False).to(device)
        relation_embeddings.weight.data *= init_size
    else:
        relation_embeddings = AttentiveEmbedding(nb_predicates=nb_predicates, nb_relations=nb_relations,
                                                 embedding_size=embedding_size, device=device).to(device)
        make_easy(predicate_lst, predicate_to_relations, relation_to_idx, relation_embeddings)

    model = NeuralKB(kernel=kernel, k=k_max).to(device)
    memory = None

    def make_hop(s: str) -> Tuple[BaseReformulator, bool]:
        nonlocal memory
        if s.isdigit():
            nb_hops, is_reversed = int(s), False
        else:
            nb_hops, is_reversed = int(s[:-1]), True
        res = None
        if reformulator_name in {'static'}:
            res = StaticReformulator(nb_hops, embedding_size)
        elif reformulator_name in {'linear'}:
            res = LinearReformulator(nb_hops, embedding_size)
        elif reformulator_name in {'attentive'}:
            res = AttentiveReformulator(nb_hops, relation_embeddings)
        elif reformulator_name in {'memory'}:
            memory = MemoryReformulator.Memory(nb_hops, nb_rules, embedding_size) if memory is None else memory
            res = MemoryReformulator(memory)
        assert res is not None
        return res, is_reversed

    hops_lst = [make_hop(s) for s in hops_str]
    hoppy = Hoppy(model=model, depth=max_depth, hops_lst=hops_lst).to(device)

    def scoring_function(story: List[Fact],
                         targets: List[Fact]) -> Tensor:
        story_rel = encode_relation(story, relation_embeddings, relation_to_idx, device)
        story_arg1, story_arg2 = encode_arguments(story, entity_embeddings, entity_to_idx, device)

        targets_rel = encode_relation(targets, relation_embeddings, relation_to_idx, device)
        targets_arg1, targets_arg2 = encode_arguments(targets, entity_embeddings, entity_to_idx, device)

        facts = [story_rel, story_arg1, story_arg2]
        scores = hoppy.score(targets_rel, targets_arg1, targets_arg2, facts)

        return scores

    def evaluate(instances: List[Instance], path: str, sample_size: Optional[int] = None) -> float:
        res = 0.0
        if len(instances) > 0:
            res = accuracy(scoring_function=scoring_function, instances=instances, sample_size=sample_size,
                           relation_to_predicate=relation_to_predicate, predicate_to_relations=predicate_to_relations)
            logger.info(f'Test Accuracy on {path}: {res:.6f}')
        return res

    loss_function = nn.BCELoss()

    N2_reg = N2() if N2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None
    entropy_reg = Entropy(use_logits=False) if entropy_weight is not None else None

    params_lst = [p for p in hoppy.parameters() if not torch.equal(p, entity_embeddings.weight)]
    params_lst += relation_embeddings.parameters()

    params = nn.ParameterList(params_lst).to(device)

    for tensor in params_lst:
        logger.info(f'\t{tensor.size()}\t{tensor.device}')

    optimizer_factory = {
        'adagrad': lambda arg: optim.Adagrad(arg, lr=learning_rate),
        'adam': lambda arg: optim.Adam(arg, lr=learning_rate),
        'sgd': lambda arg: optim.SGD(arg, lr=learning_rate)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name](params)

    global_step = 0
    hinton = HintonDiagram(max_arr=[0.0, 1.0])

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(batch_size=batch_size, nb_examples=nb_examples, nb_epochs=1, random_state=random_state)
        nb_batches = len(batcher.batches)
        epoch_loss_values = []

        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, start=1):
            global_step += 1

            indices_batch = batcher.get_batch(batch_start, batch_end)
            instances_batch = [data.train[i] for i in indices_batch]

            batch_loss_values = []

            for i, instance in enumerate(instances_batch):
                story, target = instance.story, instance.target
                s, r, o = target

                if is_debug is True and i == 0:
                    # print('STORY', story)
                    # print('TARGET', target)
                    r_lst = [r for p in predicate_lst for r in predicate_to_relations[p]]
                    r_idx_lst = [relation_to_idx[r] for r in r_lst]
                    with torch.no_grad():
                        # show_rules(model=hoppy, kernel=kernel, relation_embeddings=relation_embeddings,
                        #            data=data, relation_to_idx=relation_to_idx, device=device)
                        r_idx_tensor = torch.from_numpy(np.array(r_idx_lst, dtype=np.int64)).to(device)
                        r_tensor = relation_embeddings(r_idx_tensor)
                        k = kernel.pairwise(r_tensor, r_tensor)
                        # print(r_lst)
                        print(hinton(k.cpu().numpy()))

                story_rel = encode_relation(story, relation_embeddings, relation_to_idx, device)
                story_arg1, story_arg2 = encode_arguments(story, entity_embeddings, entity_to_idx, device)

                facts = [story_rel, story_arg1, story_arg2]

                pos_predicate = relation_to_predicate[r]
                p_relation_lst = sorted(relation_to_predicate.keys())

                target_lst = [(s, x, o) for x in p_relation_lst]
                label_lst = [int(pos_predicate == relation_to_predicate[r]) for r in p_relation_lst]

                rel_emb = encode_relation(target_lst, relation_embeddings, relation_to_idx, device)
                arg1_emb, arg2_emb = encode_arguments(target_lst, entity_embeddings, entity_to_idx, device)

                scores = hoppy.score(rel_emb, arg1_emb, arg2_emb, facts)
                labels = torch.Tensor(label_lst).float()

                # if i == 0:
                #     print(scores)
                #     print(labels)

                loss = loss_function(scores, labels)

                factors = [hoppy.factor(e) for e in [rel_emb, arg1_emb, arg2_emb]]

                loss += N2_weight * N2_reg(factors) if N2_weight is not None else 0.0
                loss += N3_weight * N3_reg(factors) if N3_weight is not None else 0.0

                if entropy_weight is not None:
                    attention = relation_embeddings.attention
                    if i == 0:
                        pass
                        # print(scores.cpu().detach().numpy())
                        # print(labels.cpu().detach().numpy())

                        # print(hinton(attention.cpu().detach().numpy()))
                        # print(attention.cpu().detach().numpy())

                    loss += entropy_weight * entropy_reg([attention])

                loss_value = loss.item()

                batch_loss_values += [loss_value]
                epoch_loss_values += [loss_value]

                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_mean, loss_std = np.mean(batch_loss_values), np.std(batch_loss_values)
            logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

            if global_step % evaluate_every == 0:
                for test_path in test_paths:
                    instances = data.test[test_path]
                    evaluate(instances=instances, path=test_path)

                    if is_debug is True:
                        for i in range(3):
                            story, target = instances[i].story, instances[i].target
                            # print('INSTANCE', target, story)

                if is_debug is True:
                    r_lst = [r for p in predicate_lst for r in predicate_to_relations[p]]
                    r_idx_lst = [relation_to_idx[r] for r in r_lst]
                    with torch.no_grad():
                        show_rules(model=hoppy, kernel=kernel, relation_embeddings=relation_embeddings,
                                   data=data, relation_to_idx=relation_to_idx, device=device)
                        r_idx_tensor = torch.from_numpy(np.array(r_idx_lst, dtype=np.int64)).to(device)
                        r_tensor = relation_embeddings(r_idx_tensor)
                        k = kernel.pairwise(r_tensor, r_tensor)
                        # print(r_lst)
                        print(hinton(k.cpu().numpy()))

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)

        slope = kernel.slope.item() if isinstance(kernel.slope, Tensor) else kernel.slope
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}\tSlope {slope:.4f}')

    for test_path in test_paths:
        evaluate(instances=data.test[test_path], path=test_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
