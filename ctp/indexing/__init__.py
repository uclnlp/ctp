# -*- coding: utf-8 -*-

from ctp.indexing.base import Index
from ctp.indexing.faiss import FAISSSearchIndex
from ctp.indexing.np import NPSearchIndex
from ctp.indexing.nms import NMSSearchIndex

__all__ = [
    'Index',
    'FAISSSearchIndex',
    'NPSearchIndex',
    'NMSSearchIndex'
]
