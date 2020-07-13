#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim

from kbcr.training.data import Data
from kbcr.training.batcher import Batcher

from kbcr.kernels import GaussianKernel

from kbcr.smart.kb import NeuralKB
from kbcr.smart.simple import SimpleHoppy

from kbcr.reformulators import BaseReformulator
from kbcr.reformulators import StaticReformulator
from kbcr.reformulators import LinearReformulator
from kbcr.reformulators import AttentiveReformulator
from kbcr.reformulators import MemoryReformulator
from kbcr.reformulators import NTPReformulator

from kbcr.regularizers import N2, N3
from kbcr.evaluation import evaluate_slow as evaluate
from kbcr.evaluation import evaluate_naive
from kbcr.evaluation import evaluate_on_countries

from typing import Tuple

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())
# torch.autograd.set_detect_anomaly(True)


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}'


def main(argv):
    parser = argparse.ArgumentParser('KBC Research', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=20)
    parser.add_argument('--k-max', '-K', action='store', type=int, default=3)

    parser.add_argument('--hops', nargs='+', type=str, default=['1', '2'])

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.001)

    parser.add_argument('--batch-size', '-b', action='store', type=int, default=8)
    parser.add_argument('--eval-batch-size', '-E', action='store', type=int, default=None)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adam',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--N2', action='store', type=float, default=None)
    parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--reformulator', '-r', action='store', type=str, default='linear',
                        choices=['static', 'linear', 'attentive', 'memory', 'ntp'])
    parser.add_argument('--nb-rules', '-R', action='store', type=int, default=4)

    parser.add_argument('--GNTP-R', action='store', type=int, default=None)

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--validate-every', '-V', action='store', type=int, default=None)
    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--init-size', '-i', action='store', type=float, default=1.0)

    parser.add_argument('--init', action='store', type=str, default='uniform')
    parser.add_argument('--ref-init', action='store', type=str, default='uniform')

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)

    parser.add_argument('--nb-negatives', action='store', type=int, default=1)

    parser.add_argument('--quiet', '-q', action='store_true', default=False)

    parser.add_argument('--freeze-entities', '-f', action='store', type=int, default=None)
    parser.add_argument('--refresh-interval', '--refresh', action='store', type=int, default=None)
    parser.add_argument('--index-type', '--index', action='store', type=str, default='faiss',
                        choices=['np', 'faiss', 'nms'])

    parser.add_argument('--lower-bound', '--lb', action='store', type=float, default=-1.0)
    parser.add_argument('--upper-bound', '--ub', action='store', type=float, default=1.0)

    parser.add_argument('--slow-eval', action='store_true', default=False)

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
    optimizer_name = args.optimizer

    N2_weight = args.N2
    N3_weight = args.N3

    reformulator_type = args.reformulator
    nb_rules = args.nb_rules

    gntp_R = args.GNTP_R

    eval_batch_size = batch_size if args.eval_batch_size is None else args.eval_batch_size

    seed = args.seed

    validate_every = args.validate_every
    input_type = args.input_type
    init_size = args.init_size

    init_type = args.init
    ref_init_type = args.ref_init

    load_path = args.load
    save_path = args.save

    nb_neg = args.nb_negatives
    is_quiet = args.quiet

    freeze_entities = args.freeze_entities
    refresh_interval = args.refresh_interval
    index_type = args.index_type

    lower_bound = args.lower_bound
    upper_bound = args.upper_bound

    slow_eval = args.slow_eval

    evaluate_ = evaluate_naive if slow_eval else evaluate

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

    entity_embeddings = nn.Embedding(data.nb_entities, embedding_size, sparse=False)
    predicate_embeddings = nn.Embedding(data.nb_predicates, embedding_size, sparse=False)

    if init_type in {'uniform'}:
        nn.init.uniform_(entity_embeddings.weight, lower_bound, upper_bound)
        nn.init.uniform_(predicate_embeddings.weight, lower_bound, upper_bound)

    nn.init.uniform_(entity_embeddings.weight, lower_bound, upper_bound)
    nn.init.uniform_(predicate_embeddings.weight, lower_bound, upper_bound)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    if freeze_entities is not None:
        entity_embeddings.weight.requires_grad = False

    kernel = GaussianKernel(slope=1.0)

    fact_rel = torch.from_numpy(np.array([data.predicate_to_idx[p] for (_, p, _) in data.train_triples])).to(device)
    fact_arg1 = torch.from_numpy(np.array([data.entity_to_idx[s] for (s, _, _) in data.train_triples])).to(device)
    fact_arg2 = torch.from_numpy(np.array([data.entity_to_idx[o] for (_, _, o) in data.train_triples])).to(device)
    facts = [fact_rel, fact_arg1, fact_arg2]

    base_model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                          k=k_max, facts=facts, kernel=kernel, device=device,
                          index_type=index_type, refresh_interval=refresh_interval).to(device)

    memory = None

    def make_hop(s: str) -> Tuple[BaseReformulator, bool]:
        nonlocal memory
        if s.isdigit():
            nb_hops, is_reversed = int(s), False
        else:
            nb_hops, is_reversed = int(s[:-1]), True
        res = None
        if reformulator_type in {'static'}:
            res = StaticReformulator(nb_hops, embedding_size, init_name=ref_init_type,
                                     lower_bound=lower_bound, upper_bound=upper_bound)
        elif reformulator_type in {'linear'}:
            res = LinearReformulator(nb_hops, embedding_size, init_name=ref_init_type,
                                     lower_bound=lower_bound, upper_bound=upper_bound)
        elif reformulator_type in {'attentive'}:
            res = AttentiveReformulator(nb_hops, predicate_embeddings, init_name=ref_init_type,
                                        lower_bound=lower_bound, upper_bound=upper_bound)
        elif reformulator_type in {'memory'}:
            if memory is None:
                memory = MemoryReformulator.Memory(nb_hops, nb_rules, embedding_size, init_name=ref_init_type,
                                                   lower_bound=lower_bound, upper_bound=upper_bound)
            res = MemoryReformulator(memory)
        elif reformulator_type in {'ntp'}:
            res = NTPReformulator(nb_hops=nb_hops, embedding_size=embedding_size,
                                  kernel=kernel, init_name=ref_init_type,
                                  lower_bound=lower_bound, upper_bound=upper_bound)
        assert res is not None
        return res, is_reversed

    hops_lst = [make_hop(s) for s in hops_str]

    # model = MultiHoppy(model=base_model, entity_embeddings=entity_embeddings, hops_lst=hops_lst).to(device)
    model = SimpleHoppy(model=base_model, entity_embeddings=entity_embeddings, hops_lst=hops_lst).to(device)

    def scoring_function(batch_xs: np.ndarray,
                         batch_xp: np.ndarray,
                         batch_xo: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_xs = torch.from_numpy(batch_xs).to(device)
            tensor_xp = torch.from_numpy(batch_xp).to(device)
            tensor_xo = torch.from_numpy(batch_xo).to(device)
            tensor_xs_emb = entity_embeddings(tensor_xs)
            tensor_xp_emb = predicate_embeddings(tensor_xp)
            tensor_xo_emb = entity_embeddings(tensor_xo)
            scores_ = model.score(tensor_xp_emb, tensor_xs_emb, tensor_xo_emb)
        return scores_.cpu().numpy()

    params_lst = {p for p in model.parameters()} | {entity_embeddings.weight, predicate_embeddings.weight}
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

    # loss_function = nn.BCELoss(reduction="sum")
    loss_function = nn.BCELoss()

    N2_reg = N2() if N2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(data, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        if freeze_entities is not None and epoch_no > freeze_entities:
            entity_embeddings.weight.requires_grad = True

        epoch_loss_values = []
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            xp_batch_np, xs_batch_np, xo_batch_np, xi_batch_np = batcher.get_batch(batch_start, batch_end)
            t = xp_batch_np.shape[0]

            assert nb_neg > 0

            xp_exp_np = np.repeat(xp_batch_np, nb_neg * 3 + 1)
            xs_exp_np = np.repeat(xs_batch_np, nb_neg * 3 + 1)
            xo_exp_np = np.repeat(xo_batch_np, nb_neg * 3 + 1)
            xi_exp_np = np.repeat(xi_batch_np, nb_neg * 3 + 1)

            xt_exp_np = np.zeros_like(xp_exp_np)
            xt_exp_np[0::nb_neg * 3 + 1] = 1

            for i in range(t):
                a_ = rs.permutation(data.nb_entities)
                b_ = rs.permutation(data.nb_entities)

                c_ = rs.permutation(data.nb_entities)
                d_ = rs.permutation(data.nb_entities)

                while a_.shape[0] < nb_neg:
                    a_ = np.concatenate([a_, rs.permutation(data.nb_entities)])
                    b_ = np.concatenate([b_, rs.permutation(data.nb_entities)])

                    c_ = np.concatenate([c_, rs.permutation(data.nb_entities)])
                    d_ = np.concatenate([d_, rs.permutation(data.nb_entities)])

                a = a_[:nb_neg]
                b = b_[:nb_neg]
                c = c_[:nb_neg]
                d = d_[:nb_neg]

                xs_exp_np[(i * nb_neg * 3) + i + 1:(i * nb_neg * 3) + nb_neg + i + 1] = a
                xo_exp_np[(i * nb_neg * 3) + nb_neg + i + 1:(i * nb_neg * 3) + nb_neg * 2 + i + 1] = b

                xs_exp_np[(i * nb_neg * 3) + nb_neg * 2 + i + 1:(i * nb_neg * 3) + nb_neg * 3 + i + 1] = c
                xo_exp_np[(i * nb_neg * 3) + nb_neg * 2 + i + 1:(i * nb_neg * 3) + nb_neg * 3 + i + 1] = d

            xp_batch = torch.from_numpy(xp_exp_np.astype('int64')).to(device)
            xs_batch = torch.from_numpy(xs_exp_np.astype('int64')).to(device)
            xo_batch = torch.from_numpy(xo_exp_np.astype('int64')).to(device)
            xi_batch = torch.from_numpy(xi_exp_np.astype('int64')).to(device)
            xt_batch = torch.from_numpy(xt_exp_np.astype('int64')).float().to(device)

            # Disable masking
            # xi_batch = None

            xp_batch_emb = predicate_embeddings(xp_batch)
            xs_batch_emb = entity_embeddings(xs_batch)
            xo_batch_emb = entity_embeddings(xo_batch)

            factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

            scores = model.score(xp_batch_emb, xs_batch_emb, xo_batch_emb, mask_indices=xi_batch)
            # scores = base_model.score(xp_batch_emb, xs_batch_emb, xo_batch_emb, mask_indices=xi_batch)

            # print(scores)
            loss = loss_function(scores, xt_batch)

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
            if 'countries' in train_path:
                dev_auc = evaluate_on_countries('dev', data.entity_to_idx, data.predicate_to_idx, scoring_function)
                print('Last AUC-PR (dev) {:.4f}'.format(dev_auc))

                test_auc = evaluate_on_countries('test', data.entity_to_idx, data.predicate_to_idx, scoring_function)
                print('Last AUC-PR (test) {:.4f}'.format(test_auc))
            else:
                for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
                    metrics = evaluate_(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                        test_triples=triples, all_triples=data.all_triples,
                                        entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                                        model=model, batch_size=eval_batch_size, device=device)
                    logger.info(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')

    if 'countries' in train_path:
        dev_auc = evaluate_on_countries('dev', data.entity_to_idx, data.predicate_to_idx, scoring_function)
        print('Last AUC-PR (dev) {:.4f}'.format(dev_auc))

        test_auc = evaluate_on_countries('test', data.entity_to_idx, data.predicate_to_idx, scoring_function)
        print('Last AUC-PR (test) {:.4f}'.format(test_auc))
    else:
        for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
            metrics = evaluate_(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                test_triples=triples, all_triples=data.all_triples,
                                entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                                model=model, batch_size=eval_batch_size, device=device)
            logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger('nmslib').setLevel(logging.WARNING)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
