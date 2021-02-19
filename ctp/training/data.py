# -*- coding: utf-8 -*-

import numpy as np

from typing import Optional, List, Tuple, Dict


def read_triples(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def triples_to_vectors(triples: List[Tuple[str, str, str]],
                       entity_to_idx: Dict[str, int],
                       predicate_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs = np.array([entity_to_idx[s] for (s, p, o) in triples], dtype=np.int32)
    Xp = np.array([predicate_to_idx[p] for (s, p, o) in triples], dtype=np.int32)
    Xo = np.array([entity_to_idx[o] for (s, p, o) in triples], dtype=np.int32)
    return Xs, Xp, Xo


class Data:
    def __init__(self,
                 train_path: str,
                 dev_path: Optional[str] = None,
                 test_path: Optional[str] = None,
                 test_i_path: Optional[str] = None,
                 test_ii_path: Optional[str] = None,
                 input_type: str = 'standard') -> None:

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.test_i_path = test_i_path
        self.test_ii_path = test_ii_path

        self.input_type = input_type
        assert self.input_type in {'standard', 'reciprocal'}

        self.Xi = self.Xs = self.Xp = self.Xo = None

        # Loading the dataset
        self.train_triples = read_triples(self.train_path) if self.train_path else []
        self.original_predicate_names = {p for (_, p, _) in self.train_triples}

        self.reciprocal_train_triples = None
        if self.input_type in {'reciprocal'}:
            self.reciprocal_train_triples = [(o, f'inverse_{p}', s) for (s, p, o) in self.train_triples]
            self.train_triples += self.reciprocal_train_triples

        self.dev_triples = read_triples(self.dev_path) if self.dev_path else []
        self.test_triples = read_triples(self.test_path) if self.test_path else []

        self.test_i_triples = read_triples(self.test_i_path) if self.test_i_path else []
        self.test_ii_triples = read_triples(self.test_ii_path) if self.test_ii_path else []

        self.all_triples = self.train_triples + self.dev_triples + self.test_triples

        self.entity_set = {s for (s, _, _) in self.all_triples} | {o for (_, _, o) in self.all_triples}
        self.predicate_set = {p for (_, p, _) in self.all_triples}

        self.nb_examples = len(self.train_triples)

        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(self.entity_set))}
        self.nb_entities = max(self.entity_to_idx.values()) + 1
        self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}

        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(self.predicate_set))}
        self.nb_predicates = max(self.predicate_to_idx.values()) + 1
        self.idx_to_predicate = {v: k for k, v in self.predicate_to_idx.items()}

        self.sp_to_o_lst: Dict[Tuple[int, int], List[int]] = {}
        self.po_to_s_lst: Dict[Tuple[int, int], List[int]] = {}

        for s, p, o in self.train_triples:
            s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], self.entity_to_idx[o]
            key_sp, key_po = (s_idx, p_idx), (p_idx, o_idx)

            if key_sp not in self.sp_to_o_lst:
                self.sp_to_o_lst[key_sp] = []
            self.sp_to_o_lst[key_sp] += [o_idx]

            if key_po not in self.po_to_s_lst:
                self.po_to_s_lst[key_po] = []
            self.po_to_s_lst[key_po] += [s_idx]

        self.inverse_of_idx = {}
        if self.input_type in {'reciprocal'}:
            for p in self.original_predicate_names:
                p_idx, ip_idx = self.predicate_to_idx[p], self.predicate_to_idx[f'inverse_{p}']
                self.inverse_of_idx.update({p_idx: ip_idx, ip_idx: p_idx})

        # Triples
        self.Xs, self.Xp, self.Xo = triples_to_vectors(self.train_triples, self.entity_to_idx, self.predicate_to_idx)
        self.Xi = np.arange(start=0, stop=self.Xs.shape[0], dtype=np.int32)

        assert self.Xs.shape == self.Xp.shape == self.Xo.shape == self.Xi.shape
        return
