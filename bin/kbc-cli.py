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
from kbcr.models import DistMult, ComplEx, NeuralKB

from kbcr.regularizers import N2, N3
from kbcr.evaluation import evaluate

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())


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

    # model params
    parser.add_argument('--model', '-m', action='store', type=str, default='distmult',
                        choices=['distmult', 'complex'])

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100)
    parser.add_argument('--eval-batch-size', '-B', action='store', type=int, default=None)

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--N2', action='store', type=float, default=None)
    parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--validate-every', '-V', action='store', type=int, default=None)

    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--gradient-accumulation-steps', '--gas', action='store', type=int, default=1)

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)

    parser.add_argument('--quiet', '-q', action='store_true', default=False)

    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    train_path = args.train

    dev_path = args.dev
    test_path = args.test

    test_i_path = args.test_i
    test_ii_path = args.test_ii

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size

    batch_size = args.batch_size
    eval_batch_size = batch_size if args.eval_batch_size is None else args.eval_batch_size

    nb_epochs = args.epochs
    seed = args.seed

    learning_rate = args.learning_rate

    N2_weight = args.N2
    N3_weight = args.N3

    validate_every = args.validate_every
    input_type = args.input_type

    gradient_accumulation_steps = args.gradient_accumulation_steps

    load_path = args.load
    save_path = args.save

    is_quiet = args.quiet

    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
        (data.test_i_triples, 'test-I'),
        (data.test_ii_triples, 'test-II'),
    ]

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1e-3

    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=True)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=True)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    parameters_lst = nn.ModuleDict({
        'entities': entity_embeddings,
        'predicates': predicate_embeddings
    })
    parameters_lst.to(device)

    if load_path is not None:
        parameters_lst.load_state_dict(torch.load(load_path))

    kernel = facts = None
    if model_name in {'ntpzero'}:
        kernel = GaussianKernel(slope=None)

        fact_rel = torch.from_numpy(np.array([data.predicate_to_idx[p] for (_, p, _) in data.train_triples])).to(device)
        fact_arg1 = torch.from_numpy(np.array([data.entity_to_idx[s] for (s, _, _) in data.train_triples])).to(device)
        fact_arg2 = torch.from_numpy(np.array([data.entity_to_idx[o] for (_, _, o) in data.train_triples])).to(device)
        facts = [fact_rel, fact_arg1, fact_arg2]

    model_factory = {
        'distmult': lambda: DistMult(entity_embeddings=entity_embeddings),
        'complex': lambda: ComplEx(entity_embeddings=entity_embeddings),
        'ntpzero': lambda: NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                    kernel=kernel, facts=facts)
    }

    assert model_name in model_factory
    model = model_factory[model_name]()
    model.to(device)

    logger.info('Model state:')
    for param_tensor in parameters_lst.state_dict():
        logger.info(f'\t{param_tensor}\t{parameters_lst.state_dict()[param_tensor].size()}')

    optimizer_factory = {
        'adagrad': lambda: optim.Adagrad(parameters_lst.parameters(), lr=learning_rate),
        'adam': lambda: optim.Adam(parameters_lst.parameters(), lr=learning_rate),
        'sgd': lambda: optim.SGD(parameters_lst.parameters(), lr=learning_rate)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name]()

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    N2_reg = N2() if N2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(data, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        epoch_loss_values = []
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            xp_batch, xs_batch, xo_batch, xi_batch = batcher.get_batch(batch_start, batch_end)

            xp_batch = torch.from_numpy(xp_batch.astype('int64')).to(device)
            xs_batch = torch.from_numpy(xs_batch.astype('int64')).to(device)
            xo_batch = torch.from_numpy(xo_batch.astype('int64')).to(device)
            xi_batch = torch.from_numpy(xi_batch.astype('int64')).to(device)

            xp_batch_emb = predicate_embeddings(xp_batch)
            xs_batch_emb = entity_embeddings(xs_batch)
            xo_batch_emb = entity_embeddings(xo_batch)

            sp_scores, po_scores = model.forward(xp_batch_emb, xs_batch_emb, xo_batch_emb)
            factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

            s_loss = loss_function(sp_scores, xo_batch)
            o_loss = loss_function(po_scores, xs_batch)

            loss = s_loss + o_loss

            loss += N2_weight * N2_reg(factors) if N2_weight is not None else 0.0
            loss += N3_weight * N3_reg(factors) if N3_weight is not None else 0.0

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if batch_no % gradient_accumulation_steps == 0 or batch_no == nb_batches:
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
        torch.save(parameters_lst.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
