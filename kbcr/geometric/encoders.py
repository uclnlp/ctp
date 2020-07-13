# -*- coding: utf-8 -*-

from torch import Tensor
from torch.nn import RNN, LSTM, GRU

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2vec_encoders.cnn_highway_encoder import CnnHighwayEncoder

from allennlp.modules.seq2seq_encoders import IntraSentenceAttentionEncoder
from allennlp.modules.similarity_functions import MultiHeadedSimilarity

from typing import Tuple, List, Callable, Optional


class Seq2SeqEncoderFactory:
    def __init__(self):
        super().__init__()

    def build(self,
              name: str,
              embedding_dim: int,
              num_heads: int = 3,
              output_dim: int = 30) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
        encoder = None
        if name in {'intra'}:
            encoder = IntraSentenceAttentionEncoder(input_dim=embedding_dim, projection_dim=output_dim, combination="1,2")
        elif name in {'multihead'}:
            sim = MultiHeadedSimilarity(num_heads, embedding_dim)
            encoder = IntraSentenceAttentionEncoder(input_dim=embedding_dim, projection_dim=embedding_dim,
                                                    similarity_function=sim, num_attention_heads=num_heads,
                                                    combination="1+2")
        assert encoder is not None
        return encoder


class Seq2VecEncoderFactory:
    def __init__(self):
        super().__init__()

    def build(self,
              name: str,
              embedding_dim: int,
              hidden_size: int = 32,
              num_filters: int = 1,
              num_heads: int = 3,
              output_dim: int = 30,
              ngram_filter_sizes: Tuple = (1, 2, 3, 4, 5),
              filters: List[List[int]] = [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
              num_highway: int = 2,
              projection_dim: int = 16) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
        encoder = None
        if name in {'boe'}:
            encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_dim, averaged=True)
        elif name in {'cnn'}:
            encoder = CnnEncoder(embedding_dim=embedding_dim, num_filters=num_filters,
                                 ngram_filter_sizes=ngram_filter_sizes, output_dim=output_dim)
        elif name in {'cnnh'}:
            encoder = CnnHighwayEncoder(embedding_dim=embedding_dim, filters=filters, num_highway=num_highway,
                                        projection_dim=projection_dim, projection_location="after_cnn")
        elif name in {'rnn'}:
            rnn = RNN(input_size=embedding_dim, bidirectional=True, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(rnn)
        elif name in {'lstm'}:
            lstm = LSTM(input_size=embedding_dim, bidirectional=True, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(lstm)
        elif name in {'gru'}:
            gru = GRU(input_size=embedding_dim, bidirectional=True, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(gru)
        elif name in {'intra'}:
            intra = IntraSentenceAttentionEncoder(input_dim=embedding_dim, projection_dim=output_dim, combination="1,2")
            aggr = PytorchSeq2VecWrapper(LSTM(input_size=embedding_dim + output_dim, bidirectional=True,
                                              hidden_size=hidden_size, batch_first=True))
            encoder = lambda x, y: aggr(intra(x, y), y)
        elif name in {'multihead'}:
            sim = MultiHeadedSimilarity(num_heads, embedding_dim)
            multi = IntraSentenceAttentionEncoder(input_dim=embedding_dim, projection_dim=embedding_dim,
                                                  similarity_function=sim, num_attention_heads=num_heads,
                                                  combination="1+2")
            aggr = PytorchSeq2VecWrapper(LSTM(input_size=embedding_dim, bidirectional=True,
                                              hidden_size=hidden_size, batch_first=True))
            encoder = lambda x, y: aggr(multi(x, y), y)
        assert encoder is not None
        return encoder
