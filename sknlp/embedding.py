import os
import tempfile
import shutil
from typing import List, Union, Optional

import mxnet as mx
from mxnet.gluon import nn
import gluonnlp

from .base import BaseModel
from .vocab import Vocab
from .loss import AdaptiveSoftmax, FullSoftmax


class Embedding(nn.HybridBlock):

    def __init__(self, vocab, embed_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab = vocab
        self._embed_size = embed_size


class NonContextEmbedding(Embedding):

    def __init__(self, vocab, embed_size, **kwargs):
        super().__init__(vocab, embed_size, **kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(len(vocab), embed_size),
                init=mx.init.Uniform(0.1), grad_stype='row_sparse'
            )

    def hybrid_forward(self, F, inputs, weight):
        """
        inputs: shape(seq_length, batch_size)
        """
        return F.Embedding(
            inputs, weight, len(self._vocab),
            self._embed_size, sparse_grad=True
        )


class Elmo(Embedding):

    def __init__(
        self, vocab, embed_size, num_layers, projection_size=512,
        hidden_size=512, dropout=0, skip_connection=True, cell_clip=3,
        projection_clip=3, **kwargs
    ):
        super().__init__(vocab, embed_size, **kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(len(vocab), embed_size),
                init=mx.init.Uniform(0.1), grad_stype='row_sparse'
            )
            self.bilm = gluonnlp.model.BiLMEncoder(
                'lstmpc', num_layers, embed_size, hidden_size, dropout=dropout,
                skip_connection=skip_connection,
                projection_size=projection_size,
                cell_clip=cell_clip, proj_clip=projection_clip, prefix='bilm_'
            )

    def hybrid_forward(self, F, inputs, weight):
        """
        inputs: shape(seq_length, batch_size)
        """
        return self.bilm(F.Embedding(
            inputs, weight, len(self._vocab),
            self._embed_size, sparse_grad=True
        ))


class Token2vec(BaseModel):

    def __init__(
        self, vocab, embed_size, loss: str = 'adaptive',
        cutoffs: List[int], div_factor: int = 4,
        weight_initializer: Optional[mx.init.Initializer] = None,
        model=None, ctx=mx.cpu(), **kwargs
    ):
        super().__init__(ctx, **kwargs)
        self._vocab = vocab
        self._embed_size = embed_size
        self.meta = {
            'loss': loss,
            'embed_size': embed_size
        }
        if loss is None:
            self._loss = None
        elif loss == 'adaptive':
            self._loss = AdaptiveSoftmax(
                embed_size, len(vocab),
                cutoffs=cutoffs, div_factor=div_factor,
                weight_initializer=weight_initializer
            )
            self.meta.update({
                'cutoffs': tuple(cutoffs),
                'div_factor': div_factor,
            })
        else:
            self._loss = FullSoftmax(
                len(vocab), weight_initializer=weight_initializer
            )
        if model is None:
            self._model = NonContextEmbedding(
                self._vocab, self._embed_size, prefix='embed_'
            )
        else:
            self._model = model
        self.meta['prefix'] = self._model.prefix

    def _build(self):
        """
        Implement this function to build.
        """
        self._trainable = {'model': self._model}
        if self._loss is not None:
            self._trainable.update({'loss': self._loss})

    def _initialize(self):
        self._model.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        self._model.hybridize()
        if self._loss is None:
            self._loss.initialize(init=mx.init.Xavier(), ctx=self._ctx)
            self._loss.hybridize()

    def _batchify_fn(self):
        return BPTTBatchify(
            self._vocab.padding_token,
            self._vocab.bos_token,
            self._vocab.eos_toke
        )

    def fit(
        self, train_dataset=None, valid_dataset=None,
        batch_size=32, last_batch='keep', n_epochs=15, optimizer='adam',
        lr: float = 3e-4, update_steps_lr: int = 300, factor: float = 0.9,
        stop_factor_lr: float = 2e-6, clip=5.0, checkpoint=None,
        save_frequency=1
    ):
        """
        Fit model.

        Parameters:
        ----
        train_dataset: list of tuples
          Each tuple is a (text, tags) pair.
        valid_dataset: list of tuples
          Each tuple is a (text, tags) pair. If None, valid log will be ignored
        cut_func: function
          Function used to segment text.
        n_epochs: int
          Number of training epochs
        optimizer: str
          Optimizers in mxnet.
        lr: float
          Start learning rate.
        clip: float
          Normal clip.
        verbose:
          If true, training loss and validation score will be logged.
        checkpoint: str
          If not None, save model using `checkpoint` as prefix.
        save_frequency: int
          If checkpoint is not None, save model every `save_frequency` epochs.
        """
        if not self._trained:
            self._build()
            self._initialize()

        dataloader = self._build_dataloader(
            train_dataset, batch_size, shuffle=True, last_batch=last_batch
        )
        self._fit(
            dataloader, valid_dataset, lr=lr, n_epochs=n_epochs,
            update_steps_lr=update_steps_lr, factor=factor,
            stop_factor_lr=stop_factor_lr, optimizer=optimizer, clip=clip,
            checkpoint=checkpoint, save_frequency=save_frequency
        )

    def __call__(self, inputs):
        self.model(inputs)

    def save(self, file_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'vocab.json'), 'w') as f:
                f.write(self._vocab.to_json())
            self._model.export(os.path.join(temp_dir, 'embedding'))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def _load(cls, file_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())

            model = nn.SymbolBlock.imports(
                os.path.join(temp_dir, 'embedding-symbol.json'), ['data'],
                os.path.join(temp_dir, 'embedding-0000.params'), ctx=ctx
            )
            with open(os.path.join(temp_dir, 'vocab.json')) as f:
                meta['vocab'] = Vocab.from_json(f.read())
        return meta, model, vocab

    @classmethod
    def load(cls, file_path, update=False, ctx=mx.cpu()):
        meta, model, vocab = cls._load(file_path)
        weight = ''.join([meta['prefix'], 'weight'])
        model.params.get(weight).grad_stype = 'row_sparse'
        if not update:
            for param in model.collect_params('embed_.*').items():
                param.grad_req = 'null'
            meta['loss'] = None
        ins = cls(**meta)
        ins._model = model
        ins._trained = True
        ins._build()
        ins._initialize()
        return ins


class Token2vecElmo(Token2vec):

    def __init__(
        self, vocab, embed_size, num_layers, projection_size=512,
        hidden_size=512, dropout=0, skip_connection=True, cell_clip=3,
        projection_clip=3, loss: str = 'adaptive',
        cutoffs: List[int], div_factor: int = 4,
        weight_initializer: Optional[mx.init.Initializer] = None,
        model=None, ctx=mx.cpu(), **kwargs
    ):
        model = Elmo(
            vocab, embed_size, projection_size=projection_size,
            hidden_size=hidden_size, dropout=dropout,
            skip_connection=skip_connection, cell_clip=cell_clip,
            projection_clip=projection_clip, prefix='embed_'
        )
        super().__init__(
            vocab, embed_size, loss=loss, cutoffs=cutoffs,
            div_factor=div_factor, weight_initializer=weight_initializer,
            model=model, ctx=ctx, **kwargs
        )
        self.meta.update({
            'projection_size': projection_size,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'skip_connection': skip_connection,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip
        })
