import mxnet as mx
from mxnet.gluon import nn


class EmbeddingLayer(nn.HybridBlock):

    def __init__(self, vocab_size, embed_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(vocab_size, embed_size),
                init=mx.init.Uniform(0.1), grad_stype='row_sparse')

    def hybrid_forward(self, F, inputs, weight):
        """
        inputs: shape(seq_length, batch_size)
        """
        return F.Embedding(inputs, weight,
                           self._vocab_size, self._embed_size,
                           sparse_grad=True)
