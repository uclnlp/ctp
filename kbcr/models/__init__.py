# # -*- coding: utf-8 -*-

from kbcr.models.base import BaseLatentFeatureModel

from kbcr.models.distmult import DistMult
from kbcr.models.complex import ComplEx
from kbcr.models.kb import NeuralKB
from kbcr.models.multi import Multi

__all__ = [
    'BaseLatentFeatureModel',
    'DistMult',
    'ComplEx',
    'NeuralKB',
    'Multi'
]
