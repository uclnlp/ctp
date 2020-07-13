#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    nb_occurrences = 32

    non_hoppy_triples = []
    hoppy_triples = []
    hoppy_only_triples = []

    # Pattern:
    # p(a, b), q(b, c) => r(a, c)

    for i in range(nb_occurrences):
        e0, e1, e2 = f'a{i}', f'b{i}', f'c{i}'

        p_triple = (f'{e0}', 'p', f'{e1}')
        q_triple = (f'{e1}', 'q', f'{e2}')
        r_triple = (f'{e0}', 'r', f'{e2}')

        non_hoppy_triples += [p_triple, q_triple]
        hoppy_triples += [p_triple, q_triple, r_triple]
        hoppy_only_triples += [r_triple]

    with open('2-hops_non_hoppy.tsv', 'w') as f:
        f.writelines(f'{s}\t{p}\t{o}\n' for s, p, o in non_hoppy_triples)

    with open('2-hops_hoppy.tsv', 'w') as f:
        f.writelines(f'{s}\t{p}\t{o}\n' for s, p, o in hoppy_triples)

    with open('2-hops_hoppy_only.tsv', 'w') as f:
        f.writelines(f'{s}\t{p}\t{o}\n' for s, p, o in hoppy_only_triples)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
