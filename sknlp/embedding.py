import logging
import os
import tempfile
import shutil
import json
import math
from typing import List, Tuple, Union, Optional

import mxnet as mx
from mxnet.gluon import nn

from .data.sampler import BPTTBatchSampler
from .data.dataloader import PrefetchDataLoader
from .base import BaseModel
from .vocab import Vocab
from .module import BiLSTM, ConvEncoder
from .loss import AdaptiveSoftmax, ElmoLoss
from .utils.file import make_tarball


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


class TokenEmbedding(Embedding):

    def __init__(self, vocab, embed_size, **kwargs):
        super().__init__(vocab, embed_size, **kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(len(vocab), embed_size),
                init=mx.init.Uniform(1), grad_stype='row_sparse'
            )

    def hybrid_forward(self, F, input, weight):
        """
        input: shape(seq_length, batch_size)
        """
        return F.Embedding(
            input, weight, len(self._vocab), self._embed_size, sparse_grad=True
        )


class Elmo(Embedding):

    def __init__(
        self, vocab, char_embed_size, word_embed_size,
        num_filters=(25, 50, 75, 100, 125, 150),
        ngram_filter_sizes=(1, 2, 3, 4, 5, 6),
        conv_layer_activation='relu',
        max_chars_per_token=20,
        num_layers=2, projection_size=300,
        hidden_size=512, dropout=0, cell_clip=3, projection_clip=3, **kwargs
    ):
        super().__init__(vocab, char_embed_size, **kwargs)
        self._max_chars_per_token = max_chars_per_token
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(len(vocab), char_embed_size),
                init=mx.init.Uniform(1), grad_stype='row_sparse'
            )
            self.char_conv = ConvEncoder(
                char_embed_size, word_embed_size,
                num_filters=num_filters, ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation, prefix='charconv_'
            )
            self.bilm = BiLSTM(
                num_layers, projection_size, input_size=word_embed_size,
                hidden_size=hidden_size, dropout=dropout, cell_clip=cell_clip,
                return_all_layers=True, prefix='bilm_'
            )

    def hybrid_forward(self, F, input, states, mask, weight):
        """
        input: shape(seq_length, batch_size, max_chars_per_token)
        """
        char_embed = F.Embedding(
            input, weight, len(self._vocab), self._embed_size, sparse_grad=True
        )
        # (batch_size * seq_length, max_chars_per_token, embed_size)
        char_embed = char_embed.reshape((-1, self._max_chars_per_token))
        word_embed = self.char_conv(F.transpose(
            char_embed, axes=(1, 0, 2)
        ))
        out_shape_ref = input.slice_axis(axis=-1, begin=0, end=1)
        out_shape_ref = out_shape_ref.broadcast_axes(
            axis=(2,), size=(self._output_size)
        )
        lstm_input = F.transpose(
            word_embed.reshape_like(out_shape_ref), axes=(1, 0, 2)
        )
        return self.bilm(lstm_input, states, mask)

    def __call__(self, input, states=None, mask=None, **kwargs):
        if states is None:
            if isinstance(input, mx.ndarray.NDArray):
                batch_size = input.shape[1]
                states = self.bilm.begin_state(
                    batch_size=batch_size, func=mx.ndarray.zeros,
                    ctx=input.context, dtype=input.dtype
                )
            else:
                states = self.bilm.begin_state(func=mx.symbol.zeros)
        return super().__call__(input, states, mask, **kwargs)


class Token2vec(BaseModel):

    def __init__(
        self, vocab, embed_size, loss: str = 'adaptive',
        cutoffs: Tuple[int] = (100, ), div_factor: int = 4,
        model=None, ctx=None, **kwargs
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
        else:
            self.loss = AdaptiveSoftmax(
                embed_size, len(vocab), cutoffs=cutoffs, div_factor=div_factor,
            )
            self.meta.update({
                'cutoffs': tuple(cutoffs),
                'div_factor': div_factor,
            })
        self.model = model

    def _build(self, ctx, initialize=True):
        """
        Implement this function to build.
        """
        if self.model is None:
            self.model = TokenEmbedding(
                self._vocab, self._embed_size, prefix='embed_'
            )
        self.meta['prefix'] = self.model.prefix
        self._trainable = {'model': self.model}
        if initialize:
            self.model.initialize(init=mx.init.Xavier(), ctx=ctx)
        if self.loss is not None:
            if initialize:
                self.loss.initialize(init=mx.init.Xavier(), ctx=ctx)
            self._trainable.update({'loss': self.loss})
        self._hybridize()

    def _hybridize(self):
        self.model.hybridize(static_alloc=True)
        if self.loss is not None:
            self.loss.hybridize(static_alloc=True)

    def fit(
        self, train_dataset=None, valid_dataset=None, batch_size=32,
        sequence_length=20, last_batch='keep', n_epochs=15, optimizer='adam',
        lr: float = 1e-3, lr_update_factor: float = 0.9,
        lr_update_epochs: int = 5, clip: float = 1.0, checkpoint=None,
        save_frequency=1, prefetch=0, multigpu=False,
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
        self._prefetch = prefetch
        if not self._trained:
            self._build(self._ctx)

        dataloader = self._build_dataloader(
            train_dataset, batch_size, sequence_length, shuffle=True,
            last_batch=last_batch
        )
        self._fit(
            dataloader, valid_dataset, lr=lr, n_epochs=n_epochs,
            optimizer=optimizer, lr_update_factor=lr_update_factor,
            lr_update_epochs=lr_update_epochs, clip=clip,
            checkpoint=checkpoint, save_frequency=save_frequency,
            multigpu=multigpu
        )

    def _batch_loss(self, loss, *args):
        return loss / args[1].sum()

    def __call__(self, input):
        return self.model(input)

    def collect_params(self):
        return self._collect_params()

    def _train_log(self, loss):
        logger.info(f'train ppl: {round(math.exp(loss), 2)}')

    def _valid_log(self, valid_dataset):
        avg_loss = self.score(valid_dataset)
        logger.info(f'valid ppl: {round(math.exp(avg_loss), 2)}')

    def predict(self, dataset, batch_size=512):
        raise NotImplementedError('predict func is not implemented.')

    def save(self, file_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'vocab.json'), 'w') as f:
                f.write(self._vocab.to_json())
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            self.model.export(os.path.join(temp_dir, 'embedding'))
            make_tarball(file_path, temp_dir)

    @classmethod
    def _load(cls, file_path, ctx):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'tar')
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())
            model = nn.SymbolBlock.imports(
                os.path.join(temp_dir, 'embedding-symbol.json'), ['data'],
                os.path.join(temp_dir, 'embedding-0000.params'), ctx=ctx
            )
            with open(os.path.join(temp_dir, 'vocab.json')) as f:
                vocab = Vocab.from_json(f.read())
        return meta, model, vocab

    @classmethod
    def load(cls, file_path, update=False, ctx=mx.cpu()):
        if not update:
            meta, model, vocab = cls._load(file_path, ctx)
            for name, param in model.collect_params().items():
                param.grad_req = 'null'
            meta['loss'] = None
        else:
            meta, model, vocab = cls._load(file_path, mx.cpu())
            sym_model = model
            model = TokenEmbedding(
                vocab, meta['embed_size'], prefix=meta['prefix']
            )
            model.initialize(ctx=ctx)
            model.weight.set_data(
                sym_model.params.get(''.join([meta['prefix'], 'weight'])).data()
            )
            meta['loss'] = None
        ins = cls(vocab, meta['embed_size'], loss=meta['loss'], model=model)
        ins._trained = True
        ins._build(ctx, initialize=False)
        return ins


class Token2vecElmo(Token2vec):

    def __init__(
        self, vocab, char_embed_size, word_embed_size,
        num_filters=(32, 32, 64, 64, 128, 128, 256),
        ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7),
        conv_layer_activation='relu',
        max_chars_per_token=20, num_layers=1, projection_size=512,
        hidden_size=512, dropout=0, cell_clip=3,
        projection_clip=3, loss: str = 'adaptive',
        cutoffs: Tuple[int] = (100, ), div_factor: int = 4,
        model=None, ctx=None, **kwargs
    ):
        model = Elmo(
            vocab, char_embed_size, word_embed_size,
            num_filters=num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
            conv_layer_activation=conv_layer_activation,
            max_chars_per_token=max_chars_per_token,
            num_layers=num_layers,
            projection_size=projection_size,
            hidden_size=hidden_size, dropout=dropout,
            cell_clip=cell_clip,
            projection_clip=projection_clip, prefix='embed_'
        )
        super().__init__(
            vocab, word_embed_size, loss=loss, cutoffs=cutoffs,
            div_factor=div_factor, model=model, ctx=ctx, **kwargs
        )
        if loss is None:
            self.loss = None
        else:
            loss_func = AdaptiveSoftmax(
                word_embed_size, len(vocab),
                cutoffs=cutoffs, div_factor=div_factor,
            )
            self.meta.update({
                'cutoffs': tuple(cutoffs),
                'div_factor': div_factor,
            })
            self.loss = ElmoLoss(loss_func)
        self.meta.update({
            'char_embed_size': char_embed_size,
            'word_embed_size': word_embed_size,
            'num_filters': num_filters,
            'ngram_filter_sizes': ngram_filter_sizes,
            'conv_layer_activation': conv_layer_activation,
            'max_chars_per_token': max_chars_per_token,
            'num_layers': num_layers,
            'projection_size': projection_size,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip
        })

    def _calculate_loss(
        self, states, input, mask, forward_labels, backward_labels,
    ):
        total_batch_size = input.shape[0] * input.shape[1] * 2
        out, states = self.model(input, states, mask)
        return self.loss(
            out, mask,
            forward_labels.astype(dtype='float32'),
            backward_labels.astype(dtype='float32'),
            mx.nd.arange(total_batch_size, ctx=input.context)
        ), states

    def _calculate_oneway_loss(
        self, states, input, mask, forward_labels, backward_labels
    ):
        return self.model.valid(input, states, mask, forward_labels)

    def _forward(self, func, one_batch):
        ctx = self._ctx
        return func(
            self.states, *[mx.nd.array(element, ctx) for element in one_batch]
        )

    def _forward_backward(self, one_batch, grad=True):
        with mx.autograd.record():
            loss, states = self._forward(self._calculate_loss, one_batch)
            self.states = _detach(states)
        if grad:
            loss.backward()
        return loss.sum().asscalar()

    def _before_epoch(self, *arg, **kwargs):
        super()._before_epoch(*arg, **kwargs)
        dataloader = kwargs['dataloader']
        self.states = self.model.bilm.begin_state(
            batch_size=dataloader.batch_size,
            func=mx.ndarray.zeros, ctx=self._ctx
        )

    def _build_dataloader(
        self, dataset, batch_size, sequence_length,
        shuffle=True, last_batch='keep'
    ):
        vocab = self._vocab
        batch_sampler = BPTTBatchSampler(
            dataset, batch_size, sequence_length,
            vocab[vocab.bos_token], vocab[vocab.eos_token],
            vocab[vocab.padding_token],
            sampler='random' if shuffle else 'sequential',
            last_batch=last_batch
        )
        if self._prefetch > 0:
            return PrefetchDataLoader(batch_sampler, batch_size)
        else:
            return batch_sampler

    def score(self, dataset, sequence_length=20, batch_size=64):
        assert self._trained
        dataloader = self._build_dataloader(
            dataset, batch_size, sequence_length, False, 'keep'
        )
        total_loss = 0
        total_word = 0
        self._before_epoch(dataloader=dataloader)
        for one_batch in dataloader:
            loss, states = self._forward(self._calculate_loss, one_batch)
            total_loss += loss.sum().asscalar()
            self.states = _detach(states)
            total_word += one_batch[1].sum()
        dataloader.reset()
        return total_loss / total_word
