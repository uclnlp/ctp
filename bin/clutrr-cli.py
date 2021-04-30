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

from ctp.util import make_batches
from ctp.clutrr import Fact, Data, Instance, accuracy

from ctp.clutrr.models import BatchNeuralKB, BatchHoppy

from ctp.reformulators import BaseReformulator
from ctp.reformulators import StaticReformulator
from ctp.reformulators import LinearReformulator
from ctp.reformulators import AttentiveReformulator
from ctp.reformulators import MemoryReformulator
from ctp.reformulators import NTPReformulator

from ctp.kernels import BaseKernel, GaussianKernel
from ctp.regularizers import N2, N3, Entropy

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
               relation_to_idx: Dict[str, int],
               device: Optional[torch.device] = None):
    idx_to_relation = {i: r for r, i in relation_to_idx.items()}

    rel_idx_pair_lst = sorted(relation_to_idx.items(), key=lambda kv: kv[1])

    for r, i in rel_idx_pair_lst:
        indices = torch.tensor([i], dtype=torch.long, device=device)

        r_emb = relation_embeddings(indices)

        hops_lst = [p for p in model.hops_lst]

        for reformulator, _ in hops_lst:
            def _to_pair(hop: Tensor) -> Tuple[str, float]:
                idx, score = decode(hop, kernel, relation_embeddings)
                rel = idx_to_relation[idx]
                return rel, score

            hop_tensor_lst = [hop for hop in reformulator(r_emb)]

            r_hops = [_to_pair(hop) for hop in hop_tensor_lst]
            print(r, ' ← ', ', '.join(f'({a} {b:.4f})' for a, b in r_hops))

            if isinstance(model.model, BatchHoppy):
                for _r_emb in hop_tensor_lst:
                    _hops_lst = [p for p in model.model.hops_lst]

                    for j, (_reformulator, _) in enumerate(_hops_lst):
                        _hop_tensor_lst = [_hop for _hop in _reformulator(_r_emb)]
                        _r_hops = [_to_pair(_hop) for _hop in _hop_tensor_lst]
                        print(j, ' ← ', ', '.join(f'({_a} {_b:.4f})' for _a, _b in _r_hops))

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
    indices = torch.tensor(indices_np, dtype=torch.long, device=device)
    res = F.embedding(indices, relation_embeddings)
    return res


def encode_arguments(facts: List[Fact],
                     entity_embeddings: Tensor,
                     entity_to_idx: Dict[str, int],
                     device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    indices_np = np.array([[entity_to_idx[s], entity_to_idx[o]] for s, _, o in facts], dtype=np.int64)
    indices = torch.tensor(indices_np, dtype=torch.long, device=device)
    emb = F.embedding(indices, entity_embeddings)
    return emb[:, 0, :], emb[:, 1, :]


def encode_entities(facts: List[Fact],
                    entity_embeddings: Tensor,
                    entity_to_idx: Dict[str, int],
                    device: Optional[torch.device]) -> Tensor:
    indices_lst = sorted({entity_to_idx[e] for s, r, o in facts for e in {s, o}})
    indices = torch.tensor(indices_lst, dtype=torch.long, device=device)
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
    argparser.add_argument('--encoder', nargs='+', type=str, default=None)

    # training params
    argparser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    argparser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)
    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=8)
    argparser.add_argument('--test-batch-size', '--tb', action='store', type=int, default=None)

    argparser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                           choices=['adagrad', 'adam', 'sgd'])

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--evaluate-every', '-V', action='store', type=int, default=1)
    argparser.add_argument('--evaluate-every-batches', action='store', type=int, default=None)

    argparser.add_argument('--N2', action='store', type=float, default=None)
    argparser.add_argument('--N3', action='store', type=float, default=None)
    argparser.add_argument('--entropy', '-E', action='store', type=float, default=None)

    argparser.add_argument('--scoring-type', '-s', action='store', type=str, default='concat',
                           choices=['concat', 'min'])

    argparser.add_argument('--tnorm', '-t', action='store', type=str, default='min',
                           choices=['min', 'prod', 'mean'])
    argparser.add_argument('--reformulator', '-r', action='store', type=str, default='linear',
                           choices=['static', 'linear', 'attentive', 'memory', 'ntp'])
    argparser.add_argument('--nb-rules', '-R', action='store', type=int, default=4)

    argparser.add_argument('--gradient-accumulation-steps', action='store', type=int, default=1)

    argparser.add_argument('--GNTP-R', action='store', type=int, default=None)

    argparser.add_argument('--slope', '-S', action='store', type=float, default=None)
    argparser.add_argument('--init-size', '-i', action='store', type=float, default=1.0)

    argparser.add_argument('--init', action='store', type=str, default='uniform')
    argparser.add_argument('--ref-init', action='store', type=str, default='random')

    argparser.add_argument('--fix-relations', '--FR', action='store_true', default=False)
    argparser.add_argument('--start-simple', action='store', type=int, default=None)

    argparser.add_argument('--debug', '-D', action='store_true', default=False)

    argparser.add_argument('--load', action='store', type=str, default=None)
    argparser.add_argument('--save', action='store', type=str, default=None)

    argparser.add_argument('--predicate', action='store_true', default=False)

    args = argparser.parse_args(argv)

    train_path = args.train
    test_paths = args.test

    embedding_size = args.embedding_size

    k_max = args.k_max
    max_depth = args.max_depth
    test_max_depth = args.test_max_depth

    hops_str = args.hops
    encoder_str = args.encoder

    nb_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    test_batch_size = batch_size if args.test_batch_size is None else args.test_batch_size

    optimizer_name = args.optimizer

    seed = args.seed

    evaluate_every = args.evaluate_every
    evaluate_every_batches = args.evaluate_every_batches

    N2_weight = args.N2
    N3_weight = args.N3
    entropy_weight = args.entropy

    scoring_type = args.scoring_type
    tnorm_name = args.tnorm
    reformulator_name = args.reformulator
    nb_rules = args.nb_rules

    nb_gradient_accumulation_steps = args.gradient_accumulation_steps

    gntp_R = args.GNTP_R

    slope = args.slope
    init_size = args.init_size

    init_type = args.init
    ref_init_type = args.ref_init

    is_fixed_relations = args.fix_relations
    start_simple = args.start_simple

    is_debug = args.debug

    load_path = args.load
    save_path = args.save

    is_predicate = args.predicate

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    data = Data(train_path=train_path, test_paths=test_paths)
    entity_lst, relation_lst = data.entity_lst, data.relation_lst
    predicate_lst = data.predicate_lst

    relation_to_predicate = data.relation_to_predicate

    test_relation_lst = ["aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law", "granddaughter",
                         "grandfather", "grandmother", "grandson", "mother", "mother-in-law", "nephew", "niece",
                         "sister", "son", "son-in-law", "uncle"]

    test_predicate_lst = sorted({relation_to_predicate[r] for r in test_relation_lst})

    nb_entities = len(entity_lst)
    nb_relations = len(relation_lst)
    nb_predicates = len(predicate_lst)

    entity_to_idx = {e: i for i, e in enumerate(entity_lst)}
    relation_to_idx = {r: i for i, r in enumerate(relation_lst)}
    predicate_to_idx = {p: i for i, p in enumerate(predicate_lst)}

    kernel = GaussianKernel(slope=slope)

    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True).to(device)
    nn.init.uniform_(entity_embeddings.weight, -1.0, 1.0)
    entity_embeddings.requires_grad = False

    relation_embeddings = nn.Embedding(nb_relations if not is_predicate else nb_predicates,
                                       embedding_size, sparse=True).to(device)

    if is_fixed_relations is True:
        relation_embeddings.requires_grad = False

    if init_type in {'uniform'}:
        nn.init.uniform_(relation_embeddings.weight, -1.0, 1.0)

    relation_embeddings.weight.data *= init_size

    model = BatchNeuralKB(kernel=kernel, scoring_type=scoring_type).to(device)
    memory: Dict[int, MemoryReformulator.Memory] = {}

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
            if nb_hops not in memory:
                memory[nb_hops] = MemoryReformulator.Memory(nb_hops, nb_rules, embedding_size, init_name=ref_init_type)
            res = MemoryReformulator(memory[nb_hops])
        elif reformulator_name in {'ntp'}:
            res = NTPReformulator(nb_hops=nb_hops, embedding_size=embedding_size,
                                  kernel=kernel, init_name=ref_init_type)
        assert res is not None
        return res.to(device), is_reversed

    hops_lst = [make_hop(s) for s in hops_str]

    encoder_model = model
    if encoder_str is not None:
        encoder_lst = [make_hop(s) for s in encoder_str]
        encoder_model = BatchHoppy(model=model, k=k_max, depth=1, tnorm_name=tnorm_name,
                                   hops_lst=encoder_lst, R=gntp_R).to(device)

    hoppy = BatchHoppy(model=encoder_model, k=k_max, depth=max_depth, tnorm_name=tnorm_name,
                       hops_lst=hops_lst, R=gntp_R).to(device)

    def scoring_function(instances_batch: List[Instance],
                         relation_lst: List[str],
                         is_train: bool = False,
                         _depth: Optional[int] = None) -> Tuple[Tensor, List[Tensor]]:

        rel_emb_lst: List[Tensor] = []
        arg1_emb_lst: List[Tensor] = []
        arg2_emb_lst: List[Tensor] = []

        story_rel_lst: List[Tensor] = []
        story_arg1_lst: List[Tensor] = []
        story_arg2_lst: List[Tensor] = []

        embeddings_lst: List[Tensor] = []

        label_lst: List[int] = []

        for i, instance in enumerate(instances_batch):

            if is_predicate is True:
                def _convert_fact(fact: Fact) -> Fact:
                    _s, _r, _o = fact
                    return _s, relation_to_predicate[_r], _o

                new_story = [_convert_fact(f) for f in instance.story]
                new_target = _convert_fact(instance.target)
                instance = Instance(new_story, new_target, instance.nb_nodes)

            story, target = instance.story, instance.target
            s, r, o = target

            story_rel = encode_relation(story, relation_embeddings.weight,
                                        predicate_to_idx if is_predicate else relation_to_idx, device)
            story_arg1, story_arg2 = encode_arguments(story, entity_embeddings.weight, entity_to_idx, device)

            embeddings = encode_entities(story, entity_embeddings.weight, entity_to_idx, device)

            target_lst: List[Tuple[str, str, str]] = [(s, x, o) for x in relation_lst]

            assert len(target_lst) == len(test_predicate_lst if is_predicate else test_relation_lst)

            # true_predicate = rel_to_predicate[r]
            # label_lst += [int(true_predicate == rel_to_predicate[r]) for r in relation_lst]

            label_lst += [int(tr == r) for tr in relation_lst]

            rel_emb = encode_relation(target_lst, relation_embeddings.weight,
                                      predicate_to_idx if is_predicate else relation_to_idx, device)
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

            def my_pad(_t: Tensor, pad: List[int]) -> Tensor:
                return torch.transpose(F.pad(torch.transpose(_t, 1, 2), pad=pad), 1, 2)

            res_t: Tensor = torch.cat([my_pad(t, pad=[0, max_len - lengths[i]]) for i, t in enumerate(t_lst)], dim=0)
            res_l: Tensor = torch.tensor([t.shape[1] for t in t_lst for _ in range(t.shape[0])],
                                         dtype=torch.long, device=device)
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

        if _depth is not None:
            hoppy.depth = _depth

        scores = hoppy.score(rel_emb, arg1_emb, arg2_emb, facts, nb_facts, _embeddings, nb_embeddings)

        if not is_train and test_max_depth is not None:
            hoppy.depth = max_depth_

        if _depth is not None:
            hoppy.depth = max_depth_

        return scores, [rel_emb, arg1_emb, arg2_emb]

    def evaluate(instances: List[Instance],
                 path: str,
                 sample_size: Optional[int] = None) -> float:
        res = 0.0
        if len(instances) > 0:
            res = accuracy(scoring_function=scoring_function,
                           instances=instances,
                           sample_size=sample_size,
                           relation_lst=test_predicate_lst if is_predicate else test_relation_lst,
                           batch_size=test_batch_size,
                           relation_to_predicate=relation_to_predicate if is_predicate else None,
                           debug=is_debug)
            logger.info(f'Test Accuracy on {path}: {res:.6f}')
        return res

    loss_function = nn.BCELoss()

    N2_reg = N2() if N2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None

    entropy_reg = Entropy(use_logits=False) if entropy_weight is not None else None

    params_lst = [p for p in hoppy.parameters() if not torch.equal(p, entity_embeddings.weight)]

    if is_fixed_relations is False:
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

        training_set, is_simple = data.train, False
        if start_simple is not None and epoch_no <= start_simple:
            training_set = [ins for ins in training_set if len(ins.story) == 2]
            is_simple = True
            logger.info(f'{len(data.train)} → {len(training_set)}')

        batcher = Batcher(batch_size=batch_size, nb_examples=len(training_set), nb_epochs=1, random_state=random_state)

        nb_batches = len(batcher.batches)
        epoch_loss_values = []

        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, start=1):
            global_step += 1

            indices_batch = batcher.get_batch(batch_start, batch_end)
            instances_batch = [training_set[i] for i in indices_batch]

            if is_predicate is True:
                label_lst: List[int] = [int(relation_to_predicate[ins.target[1]] == tp)
                                        for ins in instances_batch
                                        for tp in test_predicate_lst]
            else:
                label_lst: List[int] = [int(ins.target[1] == tr) for ins in instances_batch for tr in test_relation_lst]

            labels = torch.tensor(label_lst, dtype=torch.float32, device=device)

            scores, query_emb_lst = scoring_function(instances_batch,
                                                     test_predicate_lst if is_predicate else test_relation_lst,
                                                     is_train=True,
                                                     _depth=1 if is_simple else None)

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

            if nb_gradient_accumulation_steps > 1:
                loss = loss / nb_gradient_accumulation_steps

            loss.backward()

            if nb_gradient_accumulation_steps == 1 or global_step % nb_gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.4f}')

            if evaluate_every_batches is not None:
                if global_step % evaluate_every_batches == 0:
                    for test_path in test_paths:
                        evaluate(instances=data.test[test_path], path=test_path)

        if epoch_no % evaluate_every == 0:
            for test_path in test_paths:
                evaluate(instances=data.test[test_path], path=test_path)

            if is_debug is True:
                with torch.no_grad():
                    show_rules(model=hoppy, kernel=kernel, relation_embeddings=relation_embeddings,
                               relation_to_idx=predicate_to_idx if is_predicate else relation_to_idx, device=device)

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
