import functools
import itertools

import os
import tempfile
import shutil
import json
import logging

import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from sklearn.metrics import precision_recall_fscore_support

import gluonnlp

from .base import DeepSupervisedModel
from .data import Pad, InMemoryDataset, SequenceTagDataset
from .utils.array import sequence_mask
from .embedding import Token2vec
from .crf import Crf, viterbi_decode
from .encode import TextRNN

logger = logging.getLogger(__name__)


def batchify(input_padding, label_padding, one_batch):
    (inputs, length), labels = gluonnlp.data.batchify.Tuple(
        Pad(axis=0, pad_val=input_padding, ret_length=True),
        Pad(axis=0, pad_val=label_padding)
    )(one_batch)
    inputs = inputs.transpose((1, 0))
    mask = sequence_mask(np.ones_like(inputs), length.astype('int'))
    labels = labels.transpose((1, 0))
    return inputs, mask, labels


class DeepTagger(DeepSupervisedModel):

    def __init__(
        self, num_tags, encode_layer, embedding_layer=None, label2idx=None,
        vocab=None, segmenter=None, max_length=100, embed_size=100, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.encode_layer = encode_layer
        self._trained = False
        self._num_classes = num_tags
        self._label2idx = label2idx
        self._segmenter = segmenter
        self._max_length = max_length
        self._embed_size = embed_size
        self._vocab = vocab

        self.meta = {
            'num_tags': self._num_classes,
            'label2idx': self._label2idx,
            'max_length': self._max_length,
            'segmenter': self._segmenter,
            'embed_size': self._embed_size
        }

    def _build(self, ctx, initialize=True):
        if self.embedding_layer is None:
            self.embedding_layer = Token2vec(
                self._vocab, self._embed_size, loss=None, ctx=ctx
            )
        self.loss = Crf(self._num_classes, prefix='crf_')
        self.meta['crf_prefix'] = self.loss.prefix
        self._trainable = {
            'embedding': self.embedding_layer,
            'encode': self.encode_layer,
            'loss': self.loss
        }
        if initialize:
            self.embedding_layer._build(ctx, initialize=initialize)
            self.encode_layer.initialize(init=mx.init.Xavier(), ctx=ctx)
            self.loss.initialize(init=mx.init.Xavier(), ctx=ctx)
        self.encode_layer.hybridize(static_alloc=True)
        self.loss.hybridize(static_alloc=True)

    def _get_or_build_dataset(self, dataset, X, y):
        assert (X and y) or dataset
        if dataset:
            return dataset
        d = InMemoryDataset(X, y)
        return SequenceTagDataset(
            d, vocab=self._vocab, label2idx=self._label2idx,
            segmenter=self._segmenter, max_length=self._max_length
        )

    def _valid_log(self, valid_dataset):
        self._decode = self._create_decoder(
            self.loss.transitions.data().asnumpy())
        scores, avg_score = self.score(dataset=valid_dataset)
        for l, p, r, f, _ in zip(self.idx2labels(
                list(range(self._num_classes))), *scores):
            logger.info(
                f'label: {l} precision: {p}, recall: {r}, f1: {f}'
            )
        p, r, f, _ = avg_score
        logger.info(f'avg: {round(f * 100, 2)}({round(p * 100, 2)}, '
                    f'{round(r * 100, 2)})')
        return f

    def _calculate_logits(self, inputs, mask, *args):
        return self.encode_layer(self.embedding_layer(inputs), mask)

    def _calculate_loss(self, inputs, mask, labels):
        logits = self._calculate_logits(inputs, mask)
        return -self.loss(logits, labels, mask), None

    def _create_decoder(self, transitions):

        def decoder(inputs, mask=None):
            return viterbi_decode(transitions, inputs, mask=mask)

        return decoder

    def _batchify_fn(self):
        input_padding = self._vocab[self._vocab.padding_token]
        label_padding = self._label2idx['O']
        return functools.partial(batchify, input_padding, label_padding)

    def predict(
        self, X=None, dataset=None, batch_size=512, return_origin_label=True
    ):
        assert self._trained
        assert dataset or X
        if dataset is None:
            dataset = self._get_or_build_dataset(dataset, X, ['O'] * len(X))
        self.idx2labels = dataset.idx2labels
        dataloader = self._build_dataloader(dataset, batch_size, False, 'keep')

        predictions = []
        for one_batch in dataloader:
            for logits in self._forward(
                self._calculate_logits, one_batch, self._ctx, batch_axis=1
            ):
                mask = one_batch[1]
                predictions.extend(self._decode(logits.asnumpy(), mask))
        if return_origin_label:
            return [self.idx2labels(idx) for idx in predictions]
        return predictions

    def score(self, X=None, y=None, dataset=None, batch_size=512):
        assert self._trained
        dataset = self._get_or_build_dataset(dataset, X, y)
        predictions = self.predict(dataset=dataset, return_origin_label=False)
        y = list(itertools.chain(*[label for _, label in dataset]))
        predictions = list(itertools.chain(*predictions))
        return (
            precision_recall_fscore_support(
                y, predictions, labels=list(range(self._num_classes))
            ),
            precision_recall_fscore_support(
                y, predictions, labels=list(range(self._num_classes)),
                average='micro'
            )
        )

    def save(self, file_path: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.embedding_layer.save(os.path.join(temp_dir, 'embedding'))
            self.encode_layer.export(os.path.join(temp_dir, 'encode'))
            self.loss.export(os.path.join(temp_dir, 'crf_loss'))
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def _load_embedding_layer(cls, file_path, update, ctx):
        return Token2vec.load(file_path, update=update, ctx=ctx)

    @classmethod
    def load(cls, file_path, update=False, ctx=mx.cpu()):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())

            embedding_layer = cls._load_embedding_layer(
                os.path.join(temp_dir, 'embedding.tar.gz'), update, ctx
            )
            encode_layer = nn.SymbolBlock.imports(
                os.path.join(temp_dir, 'encode-symbol.json'), ['data0', 'data1'],
                os.path.join(temp_dir, 'encode-0000.params'), ctx=ctx
            )
            for name, param in encode_layer.collect_params().items():
                param.grad_req = 'null'
            loss = nn.SymbolBlock.imports(
                os.path.join(temp_dir, 'crf_loss-symbol.json'),
                ['data0', 'data1', 'data2'],
                os.path.join(temp_dir, 'crf_loss-0000.params'), ctx=ctx
            )

            ins = cls(
                meta['num_tags'], encode_layer,
                embedding_layer=embedding_layer,
                vocab=embedding_layer._vocab, label2idx=meta['label2idx'],
                segmenter=meta['segmenter'], max_length=meta['max_length'],
                embed_size=meta['embed_size'], ctx=ctx
            )
        ins._decode = ins._create_decoder(
            loss.params.get('crf_transitions').data().asnumpy()
        )
        ins._trained = True
        ins._build(ctx, initialize=False)
        return ins


class TextRNNTagger(DeepTagger):

    def __init__(
        self, num_tags, encode_layer=None, embedding_layer=None,
        label2idx=None, vocab=None, segmenter=None, max_length=100,
        embed_size=100, num_rnn_layers=1, projection_size=128,
        rnn_hidden_size=1024, cell_clip=3, projection_clip=3,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='tanh',
        dropout=0.5, ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextRNN(
                num_rnn_layers=num_rnn_layers,
                projection_size=projection_size,
                hidden_size=rnn_hidden_size,
                cell_clip=cell_clip,
                projection_clip=projection_clip,
                fc_activation=fc_activation,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
                dropout=dropout,
                dense_connection=None,
                output_size=num_tags,
                prefix='encode_'
            )
        super().__init__(
            num_tags, encode_layer, embedding_layer=embedding_layer,
            label2idx=label2idx, vocab=vocab, segmenter=segmenter,
            max_length=max_length, embed_size=embed_size, ctx=ctx, **kwargs
        )
        self.meta.update({
            'num_rnn_layers': num_rnn_layers,
            'projection_size': projection_size,
            'rnn_hidden_size': rnn_hidden_size,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip,
            'dropout': dropout,
            'dense_connection': None,
            'num_fc_layers': num_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'fc_activation': fc_activation
        })
        self.meta.update({'model_type': 'builtin-text_rnn_tagger'})
