# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import Tensor

from kbcr.clutrr.base import Instance, Fact
from kbcr.util import make_batches

from typing import Callable, List, Dict, Optional, Tuple, Any


def accuracy(scoring_function: Callable[[List[Fact], List[Fact]], Tensor],
             instances: List[Instance],
             relation_to_predicate: Dict[str, str],
             predicate_to_relations: Dict[str, List[str]],
             sample_size: Optional[int] = None) -> float:

    predicate_lst = sorted(predicate_to_relations.keys())
    if sample_size is not None:
        instances = instances[:sample_size]

    is_correct_lst = []

    for instance in instances:
        story, target = instance.story, instance.target
        s, r, o = target

        predicate = relation_to_predicate[r]
        relation_lst = [predicate_to_relations[p][0] for p in predicate_lst]
        summary_lst = [(s, r, o) for r in relation_lst]

        positive_index = predicate_lst.index(predicate)
        summary_lst[positive_index] = target

        with torch.no_grad():
            scores = scoring_function(story, summary_lst)
            scores_np = scores.cpu().numpy()

        is_correct = np.where(scores_np == np.amax(scores_np))[0][0] == positive_index

        # if not is_correct:
        #     print('ERROR', target, story)

        is_correct_lst += [is_correct]

    return np.mean(is_correct_lst).item() * 100.0


def accuracy_b(scoring_function: Callable[[List[Instance], List[str]], Tuple[Tensor, Any]],
               instances: List[Instance],
               relation_to_predicate: Dict[str, str],
               predicate_to_relations: Dict[str, List[str]],
               sample_size: Optional[int] = None,
               batch_size: Optional[int] = None) -> float:
    predicate_lst = sorted(predicate_to_relations.keys())
    relation_lst = [predicate_to_relations[p][0] for p in predicate_lst]

    if sample_size is not None:
        instances = instances[:sample_size]

    nb_instances = len(instances)

    batches = [(None, None)]
    if batch_size is not None:
        batches = make_batches(nb_instances, batch_size)

    nb_relations = len(relation_lst)

    is_correct_lst = []

    for batch_start, batch_end in batches:
        batch = instances[batch_start:batch_end]
        batch_size = len(batch)

        with torch.no_grad():
            scores, _ = scoring_function(batch, relation_lst)
            scores = scores.view(batch_size, nb_relations)
            scores_np = scores.cpu().numpy()
        predicted = np.argmax(scores_np, axis=1)

        def norm(a: str) -> str:
            return predicate_to_relations[relation_to_predicate[a]][0]

        true = np.array([relation_lst.index(norm(i.target[1])) for i in batch], dtype=predicted.dtype)
        is_correct_lst += (predicted == true).tolist()

    return np.mean(is_correct_lst).item() * 100.0
