import logging
import os
import tempfile
import shutil
import json
import math
import functools
from typing import List, Tuple, Union, Optional

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.data.dataloader import default_mp_batchify_fn
import gluonnlp

from .data import BPTTBatchify
from .base import BaseModel
from .vocab import Vocab
from .loss import AdaptiveSoftmax, FullSoftmax, ElmoLoss


logger = logging.getLogger(__name__)


def _detach(arr):
    if isinstance(arr, (tuple, list)):
        arr = [_detach(a) for a in arr]
    else:
        arr = arr.detach()
    return arr


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
                init=mx.init.Uniform(1), grad_stype='row_sparse'
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
        self, vocab, embed_size, num_layers=2, projection_size=512,
        hidden_size=512, dropout=0, skip_connection=True, cell_clip=3,
        projection_clip=3, **kwargs
    ):
        super().__init__(vocab, embed_size, **kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(len(vocab), embed_size),
                init=mx.init.Uniform(1), grad_stype='row_sparse'
            )
            self.embed_dropout = nn.Dropout(dropout)
            self.bilm = gluonnlp.model.BiLMEncoder(
                'lstmpc', num_layers, embed_size, hidden_size, dropout=dropout,
                skip_connection=skip_connection, proj_size=projection_size,
                cell_clip=cell_clip, proj_clip=projection_clip, prefix='bilm_'
            )

    def hybrid_forward(self, F, inputs, states, mask, weight):
        """
        inputs: shape(seq_length, batch_size)
        """
        embed = F.Embedding(
            inputs, weight, len(self._vocab),
            self._embed_size, sparse_grad=True
        )
        return self.bilm(
            self.embed_dropout(embed), states, F.transpose(mask, axes=(1, 0))
        )

    def __call__(self, inputs, states=None, mask=None, **kwargs):
        if states is None:
            if isinstance(inputs, mx.ndarray.NDArray):
                batch_size = inputs.shape[1]
                states = self.bilm.begin_state(
                    batch_size=batch_size, func=mx.ndarray.zeros,
                    ctx=inputs.context, dtype=inputs.dtype
                )
            else:
                states = self.bilm.begin_state(func=mx.symbol.zeros)
        return super().__call__(inputs, states, mask, **kwargs)


class Token2vec(BaseModel):

    def __init__(
        self, vocab, embed_size, loss: str = 'adaptive',
        cutoffs: Tuple[int] = (100, ), div_factor: int = 4,
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
            self.loss = None
        elif loss == 'adaptive':
            self.loss = AdaptiveSoftmax(
                embed_size, len(vocab), cutoffs=cutoffs, div_factor=div_factor,
            )
            self.meta.update({
                'cutoffs': tuple(cutoffs),
                'div_factor': div_factor,
            })
        else:
            self.loss = FullSoftmax(embed_size, len(vocab))
        self.model = model

    def _build(self, ctx, initialize=True):
        """
        Implement this function to build.
        """
        if self.model is None:
            self.model = NonContextEmbedding(
                self._vocab, self._embed_size, prefix='embed_'
            )
        self.meta['prefix'] = self.model.prefix
        self._trainable = {'model': self.model}
        if initialize:
            self.model.initialize(init=mx.init.Xavier(), ctx=mx.cpu())
        self.model.hybridize(static_alloc=True)
        if self.loss is not None:
            if initialize:
                self.loss.initialize(init=mx.init.Xavier(), ctx=ctx)
            self.loss.hybridize(static_alloc=True)
            self._trainable.update({'loss': self.loss})

    def fit(
        self, train_dataset=None, valid_dataset=None,
        batch_size=32, last_batch='keep', n_epochs=15,
        optimizer='adam', lr: float = 3e-4, update_steps_lr: int = 300,
        factor: float = 0.9, stop_factor_lr: float = 2e-6,
        clip=1.0, checkpoint=None, save_frequency=1, num_workers=1
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
            self._build(self._ctx)
        self._num_workers = num_workers

        dataloader = self._build_dataloader(
            train_dataset, batch_size, shuffle=True, last_batch=last_batch
        )
        self._fit(
            dataloader, valid_dataset, lr=lr, n_epochs=n_epochs,
            update_steps_lr=update_steps_lr, factor=factor,
            stop_factor_lr=stop_factor_lr, optimizer=optimizer, clip=clip,
            checkpoint=checkpoint, save_frequency=save_frequency
        )

    def _batch_loss(self, loss, *args):
        return loss / args[1].sum().asscalar()

    def __call__(self, inputs):
        return self.model(inputs)

    def collect_params(self):
        return self.model.collect_params()

    def _valid_log(self, valid_dataset):
        avg_loss = self.score(valid_dataset)
        logger.info(f'valid ppl: {round(math.exp(avg_loss), 2)}')

    def predict(self, dataset, batch_size=512):
        raise NotImplementedError('predict func is not implemented.')

    def save(self, file_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'vocab.json'), 'w') as f:
                f.write(self._vocab.to_json())
            self.model.export(os.path.join(temp_dir, 'embedding'))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def _load(cls, file_path, ctx):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())
            meta['prefix'] = 'embed_'
            model = nn.SymbolBlock.imports(
                os.path.join(temp_dir, 'embedding-symbol.json'), ['data'],
                os.path.join(temp_dir, 'embedding-0000.params'), ctx=ctx
            )
            with open(os.path.join(temp_dir, 'vocab.json')) as f:
                vocab = Vocab.from_json(f.read())
        return meta, model, vocab

    @classmethod
    def load(cls, file_path, update=False, ctx=mx.cpu()):
        meta, model, vocab = cls._load(file_path, ctx)
        weight = ''.join([meta['prefix'], 'weight'])
        model.params.get(weight).grad_stype = 'row_sparse'
        if not update:
            for name, param in model.collect_params('embed_.*').items():
                param.grad_req = 'null'
            meta['loss'] = None
        ins = cls(vocab, meta['embed_size'], loss=meta['loss'], model=model)
        ins._trained = True
        ins._build(ctx, initialize=False)
        return ins


def batchify(padding_token, bos_token, eos_token, one_batch):
    (inputs, length), forward_labels, backward_labels = BPTTBatchify(
        padding_token, bos_token, eos_token
    )(one_batch)
    inputs = inputs.transpose(axes=(1, 0))
    mask = mx.nd.SequenceMask(
        mx.nd.ones_like(inputs),
        sequence_length=length.astype('float32'),
        use_sequence_length=True
    )
    forward_labels = forward_labels.transpose(axes=(1, 0))
    backward_labels = backward_labels.transpose(axes=(1, 0))
    return default_mp_batchify_fn(
        (inputs, mask, forward_labels, backward_labels)
    )


class Token2vecElmo(Token2vec):

    def __init__(
        self, vocab, embed_size, num_layers=1, projection_size=512,
        hidden_size=512, dropout=0, skip_connection=True, cell_clip=3,
        projection_clip=3, loss: str = 'adaptive',
        cutoffs: Tuple[int] = (100, ), div_factor: int = 4,
        model=None, ctx=mx.cpu(), **kwargs
    ):
        model = Elmo(
            vocab, embed_size, num_layers=num_layers,
            projection_size=projection_size,
            hidden_size=hidden_size, dropout=dropout,
            skip_connection=skip_connection, cell_clip=cell_clip,
            projection_clip=projection_clip, prefix='embed_'
        )
        super().__init__(
            vocab, embed_size, loss=loss, cutoffs=cutoffs,
            div_factor=div_factor, model=model, ctx=ctx, **kwargs
        )
        if loss is None:
            self.loss = None
        else:
            if loss == 'adaptive':
                loss_func = AdaptiveSoftmax(
                    embed_size, len(vocab),
                    cutoffs=cutoffs, div_factor=div_factor,
                )
                self.meta.update({
                    'cutoffs': tuple(cutoffs),
                    'div_factor': div_factor,
                })
            else:
                loss_func = FullSoftmax(embed_size, len(vocab))
            self.loss = ElmoLoss(loss_func)
        self.meta.update({
            'num_layers': num_layers,
            'projection_size': projection_size,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'skip_connection': skip_connection,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip
        })

    def _calculate_loss(
        self, inputs, mask, forward_labels, backward_labels, states=None,
    ):
        out, states = self.model(inputs, states, mask)
        return self.loss(
            out, mask,
            forward_labels.astype(dtype='float32'),
            backward_labels.astype(dtype='float32')
        ), states

    def _batchify_fn(self):
        vocab = self._vocab
        return functools.partial(
            batchify, vocab[vocab.padding_token], vocab[vocab.bos_token],
            vocab[vocab.eos_token]
        )

    def _batch_loss(self, loss, *args):
        return loss / (args[1].sum().asscalar() - args[1].shape[1] / 2)

    def score(self, dataset, batch_size=64):
        assert self._trained
        dataloader = self._build_dataloader(dataset, batch_size, False, 'keep')
        total_loss = 0
        total_word = 0
        ctx = self._ctx
        for one_batch in dataloader:
            total_loss += self._forward_backward(one_batch, ctx)
            total_word += one_batch[1].sum().asscalar() - batch_size / 2
        return total_loss / total_word
