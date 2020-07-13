#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

from kbcr.util import make_batches
from kbcr.clutrr import Fact, Data, Instance, accuracy_b

from kbcr.clutrr.models.batch import BatchNeuralKB
from kbcr.clutrr.models.batch import BatchHoppy

from kbcr.reformulators import BaseReformulator
from kbcr.reformulators import StaticReformulator
from kbcr.reformulators import LinearReformulator
from kbcr.reformulators import AttentiveReformulator
from kbcr.reformulators import MemoryReformulator
from kbcr.reformulators import NTPReformulator

from kbcr.kernels import BaseKernel, GaussianKernel
from kbcr.regularizers import N2, N3, Entropy

from typing import List, Tuple, Dict, Optional

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=256, precision=4, suppress=True, threshold=sys.maxsize)

torch.set_num_threads(multiprocessing.cpu_count())


def decode(vector: Tensor,
           kernel: BaseKernel,
           relation_embeddings: nn.Module) -> Tuple[int, float]:
    weight = relation_embeddings.weight
    k = kernel.pairwise(vector, weight)[0, :]
    top_idx = k.argmax(dim=0).item()
    top_score = k[top_idx].item()
    return top_idx, top_score


def show_rules(model: BatchHoppy,
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
        for reformulator, _ in model.hops_lst:
            hops = reformulator(r_emb)
            p_hops = []
            for hop in hops:
                hop_idx, hop_score = decode(hop, kernel, relation_embeddings)
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
                 random_state: Optional[np.random.RandomState]):
        self.batch_size = batch_size
        self.nb_examples = nb_examples
        self.nb_epochs = nb_epochs
        self.random_state = random_state

        size = self.nb_epochs * self.nb_examples
        self.curriculum = np.zeros(size, dtype=np.int32)

        for epoch_no in range(nb_epochs):
            start, end = epoch_no * nb_examples, (epoch_no + 1) * nb_examples
            if self.random_state is not None:
                self.curriculum[start: end] = self.random_state.permutation(nb_examples)
            else:
                self.curriculum[start: end] = np.arange(nb_examples)

        self.batches = make_batches(self.curriculum.shape[0], self.batch_size)
        self.nb_batches = len(self.batches)

    def get_batch(self,
                  batch_start: int,
                  batch_end: int) -> np.ndarray:
        return self.curriculum[batch_start:batch_end]


def encode_relation(facts: List[Fact],
                    relation_embeddings: Tensor,
                    relation_to_idx: Dict[str, int],
                    device: Optional[torch.device] = None) -> Tensor:
    indices_np = np.array([relation_to_idx[r] for _, r, _ in facts], dtype=np.int64)
    indices = torch.from_numpy(indices_np)
    if device is not None:
        indices = indices.to(device)
    res = F.embedding(indices, relation_embeddings)
    return res


def encode_arguments(facts: List[Fact],
                     entity_embeddings: Tensor,
                     entity_to_idx: Dict[str, int],
                     device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    indices_np = np.array([[entity_to_idx[s], entity_to_idx[o]] for s, _, o in facts], dtype=np.int64)
    indices = torch.from_numpy(indices_np)
    if device is not None:
        indices = indices.to(device)
    emb = F.embedding(indices, entity_embeddings)
    return emb[:, 0, :], emb[:, 1, :]


def encode_entities(facts: List[Fact],
                    entity_embeddings: Tensor,
                    entity_to_idx: Dict[str, int],
                    device: Optional[torch.device]) -> Tensor:
    indices_lst = sorted({entity_to_idx[e] for s, r, o in facts for e in {s, o}})
    indices = torch.from_numpy(np.array(indices_lst, dtype=np.int64))
    if device is not None:
        indices = indices.to(device)
    emb = F.embedding(indices, entity_embeddings)
    return emb


def main(argv):
    argparser = argparse.ArgumentParser('CLUTRR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    train_path = test_path = "data/clutrr-emnlp/data_test/64.csv"

    argparser.add_argument('--train', action='store', type=str, default=train_path)
    argparser.add_argument('--test', nargs='+', type=str, default=[test_path])

    # model params
    argparser.add_argument('--embedding-size', '-k', action='store', type=int, default=20)
    argparser.add_argument('--k-max', '-m', action='store', type=int, default=10)
    argparser.add_argument('--max-depth', '-d', action='store', type=int, default=2)
    argparser.add_argument('--test-max-depth', action='store', type=int, default=None)

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

    argparser.add_argument('--scoring-type', '-s', action='store', type=str, default='concat', choices=['concat', 'min'])
    argparser.add_argument('--tnorm', '-t', action='store', type=str, default='min', choices=['min', 'prod'])
    argparser.add_argument('--reformulator', '-r', action='store', type=str, default='linear',
                           choices=['static', 'linear', 'attentive', 'memory', 'ntp'])
    argparser.add_argument('--nb-rules', '-R', action='store', type=int, default=4)

    argparser.add_argument('--GNTP-R', action='store', type=int, default=None)

    argparser.add_argument('--slope', '-S', action='store', type=float, default=None)
    argparser.add_argument('--init-size', '-i', action='store', type=float, default=1.0)

    argparser.add_argument('--init', action='store', type=str, default='uniform')
    argparser.add_argument('--ref-init', action='store', type=str, default='random')

    argparser.add_argument('--debug', '-D', action='store_true', default=False)

    argparser.add_argument('--load', action='store', type=str, default=None)
    argparser.add_argument('--save', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    train_path = args.train
    test_paths = args.test

    embedding_size = args.embedding_size

    k_max = args.k_max
    max_depth = args.max_depth
    test_max_depth = args.test_max_depth

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

    scoring_type = args.scoring_type
    tnorm_name = args.tnorm
    reformulator_name = args.reformulator
    nb_rules = args.nb_rules

    gntp_R = args.GNTP_R

    slope = args.slope
    init_size = args.init_size

    init_type = args.init
    ref_init_type = args.ref_init

    is_debug = args.debug

    load_path = args.load
    save_path = args.save

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    data = Data(train_path=train_path, test_paths=test_paths)

    rel_to_predicate = data.relation_to_predicate
    predicate_to_rel = data.predicate_to_relations
    entity_lst, predicate_lst, relation_lst = data.entity_lst, data.predicate_lst, data.relation_lst

    nb_examples = len(data.train)
    nb_entities = len(entity_lst)
    nb_relations = len(relation_lst)

    entity_to_idx = {e: i for i, e in enumerate(entity_lst)}
    relation_to_idx = {r: i for i, r in enumerate(relation_lst)}

    kernel = GaussianKernel(slope=slope)

    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=False).to(device)
    nn.init.uniform_(entity_embeddings.weight, -1.0, 1.0)
    entity_embeddings.requires_grad = False

    relation_embeddings = nn.Embedding(nb_relations, embedding_size, sparse=False).to(device)

    if init_type in {'uniform'}:
        nn.init.uniform_(relation_embeddings.weight, -1.0, 1.0)

    relation_embeddings.weight.data *= init_size

    model = BatchNeuralKB(kernel=kernel, scoring_type=scoring_type).to(device)
    memory = None

    def make_hop(s: str) -> Tuple[BaseReformulator, bool]:
        nonlocal memory
        if s.isdigit():
            nb_hops, is_reversed = int(s), False
        else:
            nb_hops, is_reversed = int(s[:-1]), True
        res = None
        if reformulator_name in {'static'}:
            res = StaticReformulator(nb_hops, embedding_size, init_name=ref_init_type)
        elif reformulator_name in {'linear'}:
            res = LinearReformulator(nb_hops, embedding_size, init_name=ref_init_type)
        elif reformulator_name in {'attentive'}:
            res = AttentiveReformulator(nb_hops, relation_embeddings, init_name=ref_init_type)
        elif reformulator_name in {'memory'}:
            if memory is None:
                memory = MemoryReformulator.Memory(nb_hops, nb_rules, embedding_size, init_name=ref_init_type)
            res = MemoryReformulator(memory)
        elif reformulator_name in {'ntp'}:
            res = NTPReformulator(nb_hops=nb_hops, embedding_size=embedding_size,
                                  kernel=kernel, init_name=ref_init_type)
        assert res is not None
        return res, is_reversed

    hops_lst = [make_hop(s) for s in hops_str]
    hoppy = BatchHoppy(model=model, k=k_max, depth=max_depth, tnorm_name=tnorm_name, hops_lst=hops_lst, R=gntp_R).to(device)

    def scoring_function(instances_batch: List[Instance],
                         relation_lst: List[str],
                         is_train: bool = False) -> Tuple[Tensor, List[Tensor]]:

        rel_emb_lst: List[Tensor] = []
        arg1_emb_lst: List[Tensor] = []
        arg2_emb_lst: List[Tensor] = []

        story_rel_lst: List[Tensor] = []
        story_arg1_lst: List[Tensor] = []
        story_arg2_lst: List[Tensor] = []

        embeddings_lst: List[Tensor] = []

        label_lst: List[int] = []

        for i, instance in enumerate(instances_batch):
            story, target = instance.story, instance.target
            s, r, o = target

            story_rel = encode_relation(story, relation_embeddings.weight, relation_to_idx, device)
            story_arg1, story_arg2 = encode_arguments(story, entity_embeddings.weight, entity_to_idx, device)

            embeddings = encode_entities(story, entity_embeddings.weight, entity_to_idx, device)

            true_predicate = rel_to_predicate[r]

            target_lst: List[Tuple[str, str, str]] = [(s, x, o) for x in relation_lst]
            label_lst += [int(true_predicate == rel_to_predicate[r]) for r in relation_lst]

            rel_emb = encode_relation(target_lst, relation_embeddings.weight, relation_to_idx, device)
            arg1_emb, arg2_emb = encode_arguments(target_lst, entity_embeddings.weight, entity_to_idx, device)

            batch_size = rel_emb.shape[0]
            fact_size = story_rel.shape[0]
            entity_size = embeddings.shape[0]

            # [B, E]
            rel_emb_lst += [rel_emb]
            arg1_emb_lst += [arg1_emb]
            arg2_emb_lst += [arg2_emb]

            # [B, F, E]
            story_rel_lst += [story_rel.view(1, fact_size, -1).repeat(batch_size, 1, 1)]
            story_arg1_lst += [story_arg1.view(1, fact_size, -1).repeat(batch_size, 1, 1)]
            story_arg2_lst += [story_arg2.view(1, fact_size, -1).repeat(batch_size, 1, 1)]

            # [B, N, E]
            embeddings_lst += [embeddings.view(1, entity_size, -1).repeat(batch_size, 1, 1)]

        def cat_pad(t_lst: List[Tensor]) -> Tuple[Tensor, Tensor]:
            lengths: List[int] = [t.shape[1] for t in t_lst]
            max_len: int = max(lengths)
            res_t: Tensor = torch.cat([F.pad(t, pad=[0, max_len - lengths[i]]) for i, t in enumerate(t_lst)], dim=0)
            res_l: Tensor = torch.tensor([t.shape[1] for t in t_lst for _ in range(t.shape[0])], dtype=torch.long)
            return res_t, res_l

        rel_emb = torch.cat(rel_emb_lst, dim=0)
        arg1_emb = torch.cat(arg1_emb_lst, dim=0)
        arg2_emb = torch.cat(arg2_emb_lst, dim=0)

        story_rel, nb_facts = cat_pad(story_rel_lst)
        story_arg1, _ = cat_pad(story_arg1_lst)
        story_arg2, _ = cat_pad(story_arg2_lst)
        facts = [story_rel, story_arg1, story_arg2]

        _embeddings, nb_embeddings = cat_pad(embeddings_lst)

        max_depth_ = hoppy.depth
        if not is_train and test_max_depth is not None:
            hoppy.depth = test_max_depth

        scores = hoppy.score(rel_emb, arg1_emb, arg2_emb, facts, nb_facts, _embeddings, nb_embeddings)

        if not is_train and test_max_depth is not None:
            hoppy.depth = max_depth_

        return scores, [rel_emb, arg1_emb, arg2_emb]

    def evaluate(instances: List[Instance], path: str, sample_size: Optional[int] = None) -> float:
        res = 0.0
        if len(instances) > 0:
            res = accuracy_b(scoring_function=scoring_function, instances=instances, sample_size=sample_size,
                             relation_to_predicate=rel_to_predicate, predicate_to_relations=predicate_to_rel,
                             batch_size=batch_size)
            logger.info(f'Test Accuracy on {path}: {res:.6f}')
        return res

    loss_function = nn.BCELoss()

    N2_reg = N2() if N2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None

    entropy_reg = Entropy(use_logits=False) if entropy_weight is not None else None

    params_lst = [p for p in hoppy.parameters() if not torch.equal(p, entity_embeddings.weight)]
    params_lst += relation_embeddings.parameters()

    params = nn.ParameterList(params_lst).to(device)

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

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

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(batch_size=batch_size, nb_examples=nb_examples, nb_epochs=1, random_state=random_state)

        nb_batches = len(batcher.batches)
        epoch_loss_values = []

        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, start=1):
            global_step += 1

            indices_batch = batcher.get_batch(batch_start, batch_end)
            instances_batch = [data.train[i] for i in indices_batch]

            label_lst: List[int] = []

            for i, instance in enumerate(instances_batch):
                story, target = instance.story, instance.target
                s, r, o = target
                true_predicate = rel_to_predicate[r]
                label_lst += [int(true_predicate == rel_to_predicate[r]) for r in relation_lst]

            scores, query_emb_lst = scoring_function(instances_batch, relation_lst, is_train=True)

            labels = torch.tensor(label_lst, dtype=torch.float32)

            loss = loss_function(scores, labels)

            factors = [hoppy.factor(e) for e in query_emb_lst]

            loss += N2_weight * N2_reg(factors) if N2_weight is not None else 0.0
            loss += N3_weight * N3_reg(factors) if N3_weight is not None else 0.0

            if entropy_weight is not None:
                for hop, _ in hops_lst:
                    attn_logits = hop.projection(query_emb_lst[0])
                    attention = torch.softmax(attn_logits, dim=1)
                    loss += entropy_weight * entropy_reg([attention])

            loss_value = loss.item()

            epoch_loss_values += [loss_value]

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.4f}')

            if global_step % evaluate_every == 0:
                for test_path in test_paths:
                    instances = data.test[test_path]
                    evaluate(instances=instances, path=test_path)

                if is_debug is True:
                    with torch.no_grad():
                        show_rules(model=hoppy, kernel=kernel, relation_embeddings=relation_embeddings,
                                   data=data, relation_to_idx=relation_to_idx, device=device)

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)

        slope = kernel.slope.item() if isinstance(kernel.slope, Tensor) else kernel.slope
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}\tSlope {slope:.4f}')

    import time
    start = time.time()

    for test_path in test_paths:
        evaluate(instances=data.test[test_path], path=test_path)

    end = time.time()
    logger.info(f'Evaluation took {end - start} seconds.')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
