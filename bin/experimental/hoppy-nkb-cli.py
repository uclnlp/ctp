#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor

from kbcr.training.data import Data
from kbcr.training.batcher import Batcher

from kbcr.kernels import GaussianKernel

from kbcr.smart import NeuralKB
from kbcr.smart import SimpleHoppy

from kbcr.reformulators import BaseReformulator
from kbcr.reformulators import StaticReformulator
from kbcr.reformulators import LinearReformulator
from kbcr.reformulators import AttentiveReformulator
from kbcr.reformulators import MemoryReformulator

from kbcr.regularizers import N2, N3
from kbcr.evaluation import evaluate

from typing import Tuple, List, Optional

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())
# torch.autograd.set_detect_anomaly(True)


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}'


def compute_bce_targets(batch_size: int,
                        nb_entities: int,
                        entries: List[Optional[List[int]]],
                        device: Optional[torch.device]) -> Tensor:
    coord_lst, value_lst = [], []
    for i, entry_lst in enumerate(entries):
        if entry_lst is not None:
            for entry in entry_lst:
                coord_lst += [[i, entry]]
                value_lst += [1.0]
    # [L, 2]
    coords = torch.LongTensor(coord_lst)
    # [L]
    values = torch.FloatTensor(value_lst)
    if device is not None:
        coords, values = coords.to(device), values.to(device)
    res = torch.sparse.FloatTensor(coords.t(), values, torch.Size([batch_size, nb_entities]))
    return res.to_dense()


def main(argv):
    parser = argparse.ArgumentParser('KBC Research', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=20)
    parser.add_argument('--k-max', '-K', action='store', type=int, default=1)
    parser.add_argument('--max-depth', '-d', action='store', type=int, default=1)

    parser.add_argument('--hops', nargs='+', type=str, default=['1', '2'])

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=8)

    parser.add_argument('--N2', action='store', type=float, default=None)
    parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--reformulator', '-r', action='store', type=str, default='linear',
                        choices=['static', 'linear', 'attentive', 'memory'])
    parser.add_argument('--nb-rules', '-R', action='store', type=int, default=4)

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--validate-every', '-V', action='store', type=int, default=None)
    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--init-size', '-i', action='store', type=float, default=1.0)

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)

    parser.add_argument('--nb-negatives', action='store', type=int, default=None)

    parser.add_argument('--quiet', '-q', action='store_true', default=False)

    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    test_i_path = args.test_i
    test_ii_path = args.test_ii

    embedding_size = args.embedding_size
    k_max = args.k_max

    hops_str = args.hops

    nb_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    N2_weight = args.N2
    N3_weight = args.N3

    reformulator_type = args.reformulator
    nb_rules = args.nb_rules

    eval_batch_size = batch_size

    seed = args.seed

    validate_every = args.validate_every
    input_type = args.input_type
    init_size = args.init_size

    load_path = args.load
    save_path = args.save

    nb_negatives = args.nb_negatives

    is_quiet = args.quiet

    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    rs = np.random.RandomState(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
        (data.test_i_triples, 'test-I'),
        (data.test_ii_triples, 'test-II'),
    ]

    rank = embedding_size
    init_size = init_size

    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=True)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=True)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    kernel = GaussianKernel(slope=1.0)

    fact_rel = torch.from_numpy(np.array([data.predicate_to_idx[p] for (_, p, _) in data.train_triples])).to(device)
    fact_arg1 = torch.from_numpy(np.array([data.entity_to_idx[s] for (s, _, _) in data.train_triples])).to(device)
    fact_arg2 = torch.from_numpy(np.array([data.entity_to_idx[o] for (_, _, o) in data.train_triples])).to(device)
    facts = [fact_rel, fact_arg1, fact_arg2]

    base_model = NeuralKB(entity_embeddings=entity_embeddings,
                          predicate_embeddings=predicate_embeddings,
                          facts=facts,
                          kernel=kernel,
                          k=k_max,
                          refresh_interval=32).to(device)

    memory = None

    def make_hop(s: str) -> Tuple[BaseReformulator, bool]:
        nonlocal memory
        if s.isdigit():
            nb_hops, is_reversed = int(s), False
        else:
            nb_hops, is_reversed = int(s[:-1]), True
        res = None
        if reformulator_type in {'static'}:
            res = StaticReformulator(nb_hops, rank)
        elif reformulator_type in {'linear'}:
            res = LinearReformulator(nb_hops, rank)
        elif reformulator_type in {'attentive'}:
            res = AttentiveReformulator(nb_hops, predicate_embeddings)
        elif reformulator_type in {'memory'}:
            memory = MemoryReformulator.Memory(nb_hops, nb_rules, rank) if memory is None else memory
            res = MemoryReformulator(memory)
        assert res is not None
        return res, is_reversed

    hops_lst = [make_hop(s) for s in hops_str]

    model = SimpleHoppy(model=base_model, entity_embeddings=entity_embeddings, hops_lst=hops_lst).to(device)

    params_lst = [p for p in model.parameters() if not torch.equal(p, entity_embeddings.weight)]
    params = nn.ParameterList(params_lst).to(device)

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    for tensor in params_lst:
        logger.info(f'\t{tensor.size()}\t{tensor.device}')

    optimizer = optim.Adagrad(params, lr=learning_rate)

    loss_function = nn.BCELoss()

    N2_reg = N2() if N2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(data, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        epoch_loss_values = []
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            xp_batch_np, xs_batch_np, xo_batch_np, xi_batch_np = batcher.get_batch(batch_start, batch_end)
            t = xp_batch_np.shape[0]

            if nb_negatives is not None:
                assert nb_negatives > 0

                xp_exp_np = np.repeat(xp_batch_np, nb_negatives + 1)
                xs_exp_np = np.repeat(xs_batch_np, nb_negatives + 1)
                xo_exp_np = np.repeat(xo_batch_np, nb_negatives + 1)
                xi_exp_np = np.repeat(xi_batch_np, nb_negatives + 1)
                xt_exp_np = np.zeros_like(xp_exp_np)
                xt_exp_np[0::nb_negatives] = 1

                for i in range(t):
                    a = rs.permutation(data.nb_entities)[:nb_negatives]
                    b = rs.permutation(data.nb_entities)[:nb_negatives]
                    xs_exp_np[(i * nb_negatives) + i + 1:(i * nb_negatives) + nb_negatives + i + 1] = a
                    xo_exp_np[(i * nb_negatives) + i + 1:(i * nb_negatives) + nb_negatives + i + 1] = b

                xp_batch = torch.from_numpy(xp_exp_np.astype('int64')).to(device)
                xs_batch = torch.from_numpy(xs_exp_np.astype('int64')).to(device)
                xo_batch = torch.from_numpy(xo_exp_np.astype('int64')).to(device)
                xi_batch = torch.from_numpy(xi_exp_np.astype('int64')).to(device)
                xt_batch = torch.from_numpy(xt_exp_np.astype('int64')).float().to(device)

                xp_batch_emb = predicate_embeddings(xp_batch)
                xs_batch_emb = entity_embeddings(xs_batch)
                xo_batch_emb = entity_embeddings(xo_batch)

                factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

                scores = model.score(xp_batch_emb, xs_batch_emb, xo_batch_emb, mask_indices=xi_batch)

                print(scores)

                if use_bce:
                    loss = loss_function(scores, xt_batch)
                else:
                    raise NotImplementedError
            else:
                xp_batch = torch.from_numpy(xp_batch_np.astype('int64')).to(device)
                xs_batch = torch.from_numpy(xs_batch_np.astype('int64')).to(device)
                xo_batch = torch.from_numpy(xo_batch_np.astype('int64')).to(device)
                xi_batch = torch.from_numpy(xi_batch_np.astype('int64')).to(device)

                xp_batch_emb = predicate_embeddings(xp_batch)
                xs_batch_emb = entity_embeddings(xs_batch)
                xo_batch_emb = entity_embeddings(xo_batch)

                sp_scores, po_scores = model.forward(xp_batch_emb, xs_batch_emb, xo_batch_emb, mask_indices=xi_batch)

                factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

                if use_bce:
                    sp_objects = [data.sp_to_o_lst.get((xs, xp), None) for xs, xp in zip(xs_batch_np, xp_batch_np)]
                    po_subjects = [data.po_to_s_lst.get((xp, xo), None) for xp, xo in zip(xp_batch_np, xo_batch_np)]

                    sp_targets = compute_bce_targets(xp_batch.shape[0], data.nb_entities, sp_objects, device=device)
                    po_targets = compute_bce_targets(xp_batch.shape[0], data.nb_entities, po_subjects, device=device)

                    s_loss = loss_function(sp_scores, sp_targets)
                    o_loss = loss_function(po_scores, po_targets)
                else:
                    s_loss = loss_function(sp_scores, xo_batch)
                    o_loss = loss_function(po_scores, xs_batch)

                loss = s_loss + o_loss

            loss += N2_weight * N2_reg(factors) if N2_weight is not None else 0.0
            loss += N3_weight * N3_reg(factors) if N3_weight is not None else 0.0

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            if not is_quiet:
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f}')

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

        if validate_every is not None and epoch_no % validate_every == 0:
            for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
                metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                   test_triples=triples, all_triples=data.all_triples,
                                   entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                                   model=model, batch_size=eval_batch_size, device=device)
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')

    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
