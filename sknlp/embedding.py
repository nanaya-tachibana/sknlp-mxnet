import os
import tempfile
import shutil
import mxnet as mx
from mxnet.gluon import nn

from .vocab import Vocab


class EmbeddingLayer(nn.HybridBlock):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hybrid_forward(self, F, inputs):
        raise NotImplementedError('forward function is not implemented.')

    def save(self, file_path):
        raise NotImplementedError('save function is not implemented.')

    @classmethod
    def load(self, file_path, ctx=mx.cpu()):
        raise NotImplementedError('load function is not implemented.')


class NonContextEmbeddingLayer(EmbeddingLayer):

    def __init__(self, vocab, embed_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab = vocab
        self._embed_size = embed_size
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(len(vocab), embed_size),
                init=mx.init.Uniform(0.1), grad_stype='row_sparse')

    def hybrid_forward(self, F, inputs, weight):
        """
        inputs: shape(seq_length, batch_size)
        """
        return F.Embedding(inputs, weight,
                           len(self._vocab), self._embed_size,
                           sparse_grad=True)

    def save(self, file_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'vocab.json'), 'w') as f:
                f.write(self._vocab.to_json())
            self.export(os.path.join(temp_dir, 'embedding'))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def load(cls, file_path, prefix='embed_', update=False, ctx=mx.cpu()):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            ins = nn.SymbolBlock.imports(
                os.path.join(temp_dir, 'embedding-symbol.json'), ['data'],
                os.path.join(temp_dir, 'embedding-0000.params'), ctx=ctx
            )
            weight = ''.join([prefix, 'weight'])
            ins.params.get(weight).grad_stype = 'row_sparse'
            if not update:
                ins.params.get(weight).grad_req = 'null'
            with open(os.path.join(temp_dir, 'vocab.json')) as f:
                ins._vocab = Vocab.from_json(f.read())
            ins.hybridize()
        return ins
