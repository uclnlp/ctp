# -*- coding: utf-8 -*-

from kbcr.indexing.base import Index
from kbcr.indexing.faiss import FAISSSearchIndex
from kbcr.indexing.np import NPSearchIndex
from kbcr.indexing.nms import NMSSearchIndex

__all__ = [
    'Index',
    'FAISSSearchIndex',
    'NPSearchIndex',
    'NMSSearchIndex'
]
