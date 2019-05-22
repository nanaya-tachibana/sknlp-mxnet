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

from .base import DeepSupervisedModel
from .data import ClassifyDataset, InMemoryDataset, Pad
from .embedding import NonContextEmbeddingLayer
from .encode import TextCNN, TextRCNN, TextRNN
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


class DeepClassifier(DeepSupervisedModel):

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
            self._loss = mx.gluon.loss.SigmoidBCELoss()
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
        return self._loss(
            self._calculate_logits(batch_inputs, batch_mask),
            batch_labels.astype(dtype='float32'),
        )

    def _batchify_fn(self):

        def batchify(one_batch):
            (batch_inputs, batch_length), batch_labels = \
                gluonnlp.data.batchify.Tuple(
                    Pad(axis=0, pad_val=self._vocab['<pad>'], ret_length=True),
                    gluonnlp.data.batchify.Stack()
            )(one_batch)
            batch_inputs = batch_inputs.transpose(axes=(1, 0))
            batch_mask = mx.nd.SequenceMask(
                mx.nd.ones_like(batch_inputs),
                sequence_length=batch_length.astype('float32'),
                use_sequence_length=True
            )
            return (
                batch_inputs.as_in_context(self._ctx),
                batch_mask.as_in_context(self._ctx),
                batch_labels.as_in_context(self._ctx)
            )

        return batchify

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
            batch_inputs, batch_mask, _ = one_batch
            logits = self._calculate_logits(batch_inputs, batch_mask)
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
        y = [label for _, label in dataset]
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
        num_filters=(25, 50, 75, 100), ngram_filter_sizes=(1, 2, 3, 4),
        conv_layer_activation='tanh', num_highway=1, num_fc_layers=2,
        fc_hidden_size=512, fc_activation='tanh', ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextCNN(
                embed_size=embed_size,
                num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                num_highway=num_highway,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
                output_size=num_classes,
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
        self._num_highway = num_highway
        self._num_fc_layers = num_fc_layers
        self._fc_hidden_size = fc_hidden_size
        self._fc_activation = fc_activation

        self.meta.update({
            'num_filters': list(self._num_filters),
            'ngram_filter_sizes': list(self._ngram_filter_sizes),
            'conv_layer_activation': self._conv_layer_activation,
            'num_highway': self._num_highway,
            'num_fc_layers': self._num_fc_layers,
            'fc_hidden_size': self._fc_hidden_size,
            'fc_activation': self._fc_activation
        })

    def _batchify_fn(self):

        def batchify(one_batch):
            (batch_inputs, batch_length), batch_labels = \
                gluonnlp.data.batchify.Tuple(
                    Pad(
                        axis=0, pad_val=self._vocab['<pad>'], ret_length=True,
                        min_length=max(self._ngram_filter_sizes)
                    ),
                    gluonnlp.data.batchify.Stack()
            )(one_batch)
            batch_inputs = batch_inputs.transpose(axes=(1, 0))
            batch_mask = mx.nd.SequenceMask(
                mx.nd.ones_like(batch_inputs),
                sequence_length=batch_length.astype('float32'),
                use_sequence_length=True
            )
            return (
                batch_inputs.as_in_context(self._ctx),
                batch_mask.as_in_context(self._ctx),
                batch_labels.as_in_context(self._ctx)
            )

        return batchify

    @classmethod
    def _load_encode_layer(cls, file_path, prefix, ctx):
        return TextCNN.load(file_path, prefix=prefix, ctx=ctx)


class TextRNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100, threshold=None,
        num_rnn_layers=1, projection_size=128, hidden_size=1024,
        cell_clip=3, projection_clip=3, dropout=0.5, dense_connection='last',
        num_fc_layers=2, fc_hidden_size=512, fc_activation='tanh',
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextRNN(
                num_rnn_layers=num_rnn_layers,
                projection_size=projection_size,
                hidden_size=hidden_size,
                cell_clip=cell_clip,
                projection_clip=projection_clip,
                fc_activation=fc_activation,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
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
        self._num_rnn_layers = num_rnn_layers
        self._projection_size = projection_size
        self._hidden_size = hidden_size
        self._cell_clip = cell_clip
        self._projection_clip = projection_clip
        self._dropout = dropout
        self._dense_connection = dense_connection
        self._num_fc_layers = num_fc_layers
        self._fc_hidden_size = fc_hidden_size
        self._fc_activation = fc_activation

        self.meta.update({
            'num_rnn_layers': self._num_rnn_layers,
            'projection_size': self._projection_size,
            'hidden_size': self._hidden_size,
            'cell_clip': self._cell_clip,
            'projection_clip': self._projection_clip,
            'dropout': self._dropout,
            'dense_connection': self._dense_connection,
            'num_fc_layers': self._num_fc_layers,
            'fc_hidden_size': self._fc_hidden_size,
            'fc_activation': self._fc_activation
        })

    @classmethod
    def _load_encode_layer(cls, file_path, prefix, ctx):
        return TextRNN.load(file_path, prefix=prefix, ctx=ctx)


class TextRCNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100, threshold=None,
        num_rnn_layers=1, projection_size=128, hidden_size=1024,
        cell_clip=3, projection_clip=3, dropout=0.5, kmax=2,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='tanh',
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextRCNN(
                num_rnn_layers=num_rnn_layers,
                projection_size=projection_size,
                hidden_size=hidden_size,
                cell_clip=cell_clip,
                projection_clip=projection_clip,
                kmax=kmax,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
                fc_activation=fc_activation,
                dropout=dropout,
                output_size=num_classes,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter,
            max_length=max_length, embed_size=embed_size,
            threshold=threshold, **kwargs)
        self._num_rnn_layers = num_rnn_layers
        self._projection_size = projection_size
        self._hidden_size = hidden_size
        self._cell_clip = cell_clip
        self._projection_clip = projection_clip
        self._kmax = kmax
        self._dropout = dropout
        self._num_fc_layers = num_fc_layers
        self._fc_hidden_size = fc_hidden_size
        self._fc_activation = fc_activation

        self.meta.update({
            'num_rnn_layers': self._num_rnn_layers,
            'projection_size': self._projection_size,
            'hidden_size': self._hidden_size,
            'cell_clip': self._cell_clip,
            'projection_clip': self._projection_clip,
            'kmax': self._kmax,
            'dropout': self._dropout,
            'num_fc_layers': self._num_fc_layers,
            'fc_hidden_size': self._fc_hidden_size,
            'fc_activation': self._fc_activation
        })

    @classmethod
    def _load_encode_layer(cls, file_path, prefix, ctx):
        return TextRCNN.load(file_path, prefix=prefix, ctx=ctx)
