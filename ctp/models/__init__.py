# # -*- coding: utf-8 -*-

from ctp.models.base import BaseLatentFeatureModel

from ctp.models.distmult import DistMult
from ctp.models.complex import ComplEx
from ctp.models.kb import NeuralKB
from ctp.models.multi import Multi

__all__ = [
    'BaseLatentFeatureModel',
    'DistMult',
    'ComplEx',
    'NeuralKB',
    'Multi'
]
