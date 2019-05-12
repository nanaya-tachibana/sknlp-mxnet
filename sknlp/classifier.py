import itertools
import json
import os
import shutil
import tempfile
import logging

import gluonnlp
import mxnet as mx
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_fscore_support

from .base import DeepModel
from .data import ClassifyDataset, InMemoryDataset, Pad
from .embedding import NonContextEmbeddingLayer
from .encode import TextCNN, TextRCNN, TextRNN
from .loss import SampledSigmoidBCELoss
from .segmenter import Segmenter

logger = logging.getLogger(__name__)


def _decode(logits, threshold, is_binary=False, is_multilabel=False):
    if is_binary and not is_multilabel:
        assert logits.shape[1] == 2
        return np.where(expit(logits[:, 1]) > threshold, 1, 0)
    elif is_multilabel:
        return np.where(expit(logits) > threshold, 1, 0)
    else:
        return np.argmax(logits, axis=1)


class DeepClassifier(DeepModel):

    def __init__(
        self, num_classes, encode_layer, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None,
        segmenter='jieba', max_length=100, embed_size=100,
        threshold=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.encode_layer = encode_layer
        self._trained = False
        self._num_classes = num_classes
        self._is_multilabel = is_multilabel
        self._segmenter = segmenter
        self._cut = Segmenter(segmenter).cut
        self._max_length = max_length
        self._embed_size = embed_size
        self._vocab = vocab
        self._label2idx = label2idx
        self._threshold = threshold or dict()

        self.meta = {
            'num_classes': self._num_classes,
            'is_multilabel': self._is_multilabel,
            'label2idx': self._label2idx,
            'max_length': self._max_length,
            'segmenter': self._segmenter,
            'embed_size': self._embed_size,
            'threshold': self._threshold
        }

    def _build(self):
        if self.embedding_layer is None:
            self.embedding_layer = NonContextEmbeddingLayer(
                self._vocab, self._embed_size, prefix='embed_'
            )
        self.meta['embedding_prefix'] = self.embedding_layer.prefix
        self.meta['encode_prefix'] = self.encode_layer.prefix
        if self._is_multilabel:
            self._loss = SampledSigmoidBCELoss()
        else:
            self._loss = mx.gluon.loss.SoftmaxCELoss(sparse_label=False)
        self._trainable = {
            'embedding': self.embedding_layer,
            'encode': self.encode_layer
        }

    def _get_or_build_dataset(self, dataset, X, y):
        assert (X and y) or dataset
        if dataset:
            return dataset
        d = InMemoryDataset(X, y)
        return ClassifyDataset(
            d, vocab=self._vocab, label2idx=self._label2idx,
            segmenter=self._cut, max_length=self._max_length
        )

    def _decode(self, x):
        threshold = np.array([
            self._threshold.get(l, 0.5)
            for l in self.idx2labels(range(self._num_classes))
        ])
        return _decode(x, threshold, is_binary=self._num_classes == 2,
                       is_multilabel=self._is_multilabel)

    def _debinarize(self, binarized_label):
        l = [
            [i for i, t in enumerate(bl) if t == 1] for bl in binarized_label
        ]
        if self._is_multilabel:
            return l
        return list(itertools.chain(*l))

    def _decode_label(self, idx):
        if isinstance(idx[0], list):
            return [self.idx2labels(i) for i in idx]
        return self.idx2labels(idx)

    def _valid_log(self, valid_dataset):
        s = self.score(dataset=valid_dataset)
        if self._is_multilabel or self._num_classes > 2:
            scores, avg_score = s
            for l, p, r, f, _ in zip(self.idx2labels(
                    list(range(self._num_classes))), *scores):
                logger.info(f'label: {l} precision: {p}, '
                            f'recall: {r}, f1: {f}')
            p, r, f, _ = avg_score
            logger.info(f'avg: {round(f * 100, 2)}({round(p * 100, 2)}, '
                        f'{round(r * 100, 2)})')
        else:
            p, r, f, _ = scores
            logger.info(f'precision: {p}, recall: {r}, f1: {f}')

    def _calculate_logits(self, batch_inputs, batch_mask):
        return self.encode_layer(
            self.embedding_layer(batch_inputs), batch_mask
        )

    def _calculate_loss(self, batch_inputs, batch_mask, batch_labels):
        batch_label_mask = mx.nd.where(
            batch_labels == 1, batch_labels, mx.nd.where(
                mx.nd.broadcast_lesser_equal(
                    mx.nd.random_uniform(shape=batch_labels.shape),
                    self.label_weights
                ),
                mx.nd.ones_like(batch_labels),
                mx.nd.zeros_like(batch_labels)
            )
        )
        return self._loss(
            self._calculate_logits(
                batch_inputs.transpose(axes=(1, 0)),
                batch_mask.transpose(axes=(1, 0))
            ),
            batch_labels.astype(dtype='float32'),
            batch_label_mask.astype(dtype='float32')
        )

    def _batchify_fn(self):
        return gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1), Pad(axis=0),
            gluonnlp.data.batchify.Stack()
        )

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
        for (batch_inputs, batch_mask, _) in dataloader:
            batch_inputs = batch_inputs.as_in_context(self._ctx)
            batch_mask = batch_mask.as_in_context(self._ctx)
            logits = self._calculate_logits(
                batch_inputs.transpose(axes=(1, 0)),
                batch_mask.transpose(axes=(1, 0))
            )
            predictions.extend(self._decode(logits.asnumpy()))
        if return_origin_label:
            if self._is_multilabel:
                predictions = self._debinarize(predictions)
            return self._decode_label(predictions)
        return predictions

    def score(self, X=None, y=None, dataset=None, batch_size=512):
        assert self._trained
        dataset = self._get_or_build_dataset(dataset, X, y)
        predictions = self.predict(dataset=dataset, return_origin_label=False)
        y = [label for _, _, label in dataset]
        if not self._is_multilabel:
            y = [np.argmax(label) for label in y]
        y = np.vstack(y)
        predictions = np.vstack(predictions)
        if self._num_classes == 2 and not self._is_multilabel:
            return precision_recall_fscore_support(
                y, predictions, average='binary'
            )
        else:
            return (
                precision_recall_fscore_support(
                    y, predictions, labels=list(range(self._num_classes))
                ),
                precision_recall_fscore_support(
                    y, predictions,
                    labels=list(range(self._num_classes)), average='micro'
                )
            )

    def save(self, file_path: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.embedding_layer.save(os.path.join(temp_dir, 'embedding'))
            self.encode_layer.save(os.path.join(temp_dir, 'encode'))
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def _load_embedding_layer(cls, file_path, prefix, update, ctx):
        return NonContextEmbeddingLayer.load(
            file_path, prefix=prefix, update=update, ctx=ctx
        )

    @classmethod
    def _load_encode_layer(cls, file_path, prefix, ctx):
        raise NotImplementedError(
            'load encode layer func is not implemented.'
        )

    @classmethod
    def load(cls, file_path, update=False, ctx=mx.cpu()):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())

            embedding_layer = cls._load_embedding_layer(
                os.path.join(temp_dir, 'embedding.tar.gz'),
                meta.pop('embedding_prefix'), update, ctx
            )
            encode_layer = cls._load_encode_layer(
                os.path.join(temp_dir, 'encode.tar.gz'),
                meta.pop('encode_prefix'), ctx
            )

            meta['ctx'] = ctx
            meta['vocab'] = embedding_layer._vocab
            meta['encode_layer'] = encode_layer
            meta['embedding_layer'] = embedding_layer
            clf = cls(**meta)
        clf._build()
        clf._trained = True
        return clf


class TextCNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100, threshold=None,
        num_filters=(25, 50, 75, 100, 125, 150),
        ngram_filter_sizes=(1, 2, 3, 4, 5, 6),
        num_highway=1, conv_layer_activation='tanh', dropout=0.5,
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextCNN(
                embed_size=embed_size,
                num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                output_size=num_classes,
                dropout=dropout,
                num_highway=num_highway,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter,
            max_length=max_length, embed_size=embed_size,
            threshold=threshold, **kwargs)
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._conv_layer_activation = conv_layer_activation
        self._dropout = dropout
        self._num_highway = num_highway

        self.meta.update({
            'num_filters': list(self._num_filters),
            'ngram_filter_sizes': list(self._ngram_filter_sizes),
            'conv_layer_activation': self._conv_layer_activation,
            'dropout': self._dropout,
            'num_highway': self._num_highway})

    def _batchify_fn(self):
        return gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1, min_length=max(self._ngram_filter_sizes)),
            Pad(axis=0, min_length=max(self._ngram_filter_sizes)),
            gluonnlp.data.batchify.Stack()
        )

    @classmethod
    def _load_encode_layer(cls, file_path, prefix, ctx):
        return TextCNN.load(file_path, prefix=prefix, ctx=ctx)


class TextRNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100, threshold=None,
        hidden_size=512, num_rnn_layers=1, output_size=1, dropout=0.5,
        dense_connection='last', ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextRNN(
                hidden_size=hidden_size,
                num_rnn_layers=num_rnn_layers,
                dropout=dropout,
                dense_connection=dense_connection,
                output_size=num_classes,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter,
            max_length=max_length, embed_size=embed_size,
            threshold=threshold, **kwargs)
        self._hidden_size = hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._dropout = dropout
        self._dense_connection = dense_connection

        self.meta.update({
            'hidden_size': hidden_size,
            'num_rnn_layers': num_rnn_layers,
            'dropout': dropout,
            'dense_connection': dense_connection
        })

    @classmethod
    def _load_encode_layer(cls, file_path, prefix, ctx):
        return TextRNN.load(file_path, prefix=prefix, ctx=ctx)


class TextRCNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100, threshold=None,
        num_filters=(25, 50, 75, 100, 125, 150),
        ngram_filter_sizes=(1, 2, 3, 4, 5, 6),
        conv_layer_activation='tanh', rnn_hidden_size=512, num_rnn_layers=1,
        dropout=0.5, num_highway=1, ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextRCNN(
                embed_size=embed_size,
                num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                output_size=num_classes,
                dropout=dropout,
                num_highway=num_highway,
                rnn_hidden_size=rnn_hidden_size,
                num_rnn_layers=num_rnn_layers,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter,
            max_length=max_length, embed_size=embed_size,
            threshold=threshold, **kwargs)
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._conv_layer_activation = conv_layer_activation
        self._dropout = dropout
        self._num_highway = num_highway
        self._rnn_hidden_size = rnn_hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._dropout = dropout

        self.meta.update({
            'num_filters': list(self._num_filters),
            'ngram_filter_sizes': list(self._ngram_filter_sizes),
            'conv_layer_activation': self._conv_layer_activation,
            'dropout': self._dropout,
            'num_highway': self._num_highway,
            'rnn_hidden_size': rnn_hidden_size,
            'num_rnn_layers': num_rnn_layers,
        })

    def _batchify_fn(self):
        return gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1, min_length=max(self._ngram_filter_sizes)),
            Pad(axis=0, min_length=max(self._ngram_filter_sizes)),
            gluonnlp.data.batchify.Stack())

    @classmethod
    def _load_encode_layer(cls, file_path, prefix, ctx):
        return TextRCNN.load(file_path, prefix=prefix, ctx=ctx)
