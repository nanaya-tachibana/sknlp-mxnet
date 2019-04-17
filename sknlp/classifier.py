import os
import json
import itertools
import tempfile
import shutil

import numpy as np
from scipy.special import expit
import mxnet as mx
from mxnet.gluon import nn, rnn

import gluonnlp
from sklearn.metrics import precision_recall_fscore_support

from .base import DeepModel
from .data import Pad, _SimpleClassifyDataset
from .embedding import EmbeddingLayer


class TextCNN(nn.HybridBlock):

    def __init__(self, embed_size=100, num_filters=(25, 50, 75, 100),
                 ngram_filter_sizes=(1, 2, 3, 4),
                 conv_layer_activation='tanh', output_size=1, dropout=0,
                 num_highway=1, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.cnn_layer = gluonnlp.model.ConvolutionalEncoder(
                embed_size=embed_size, num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                output_size=None, num_highway=num_highway,
                prefix='cnn_')
            self.cnn_dropout = nn.Dropout(
                dropout, prefix='cnndropout_')
            self.dense_layer = nn.Dense(output_size, flatten=False,
                                        prefix='dense_')

    def hybrid_forward(self, F, inputs, mask):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        return self.dense_layer(self.cnn_dropout(self.cnn_layer(inputs,
                                                                mask)))


class TextRNN(nn.HybridBlock):

    def __init__(self, hidden_size=512, num_rnn_layers=1, output_size=1,
                 dropout=0.0, dense_connection='last', **kwargs):
        super().__init__(**kwargs)
        self._dense_connection = dense_connection
        with self.name_scope():
            self.rnn_layer = rnn.LSTM(hidden_size // 2, num_rnn_layers,
                                      dropout=dropout, bidirectional=True,
                                      prefix='rnn_')
            self.rnn_dropout = nn.Dropout(
                dropout, prefix='rnndropout_')
            self.dense_layer = nn.Dense(output_size, flatten=False,
                                        prefix='dense_')

    def hybrid_forward(self, F, inputs, mask=None):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        rnn_output = self.rnn_layer(inputs)
        if self._dense_connection == 'last' and mask is not None:
            rnn_output = F.SequenceLast(rnn_output,
                                        sequence_length=F.sum(mask, axis=0),
                                        use_sequence_length=True)
        if self._dense_connection == 'average':
            rnn_output = F.sum(rnn_output, axis=0)
        return self.dense_layer(self.rnn_dropout(rnn_output))


def _decode(logits, is_binary=False, is_multilabel=False, threshold=0.5):
    if is_binary and not is_multilabel:
        assert logits.shape[1] == 2
        return np.where(expit(logits[:, 1]) > threshold, 1, 0)
    elif is_multilabel:
        return np.where(expit(logits) > threshold, 1, 0)
    else:
        return np.argmax(logits, axis=1)


class _ClassifyBlockComposition(nn.HybridBlock):

    def __init__(self, embedding_layer, encode_layer, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.embedding_layer = embedding_layer
            self.encode_layer = encode_layer

    def hybrid_forward(self, F, inputs, mask=None):
        return self.encode_layer(self.embedding_layer(inputs), mask)


class DeepClassifier(DeepModel):

    def __init__(self, num_classes=2, class_weight=None,
                 is_multilabel=False, label2idx=None, ctx=mx.cpu(),
                 vocab=None, embed_weight=None,
                 segmenter='jieba', max_length=100, embed_size=100, **kwargs):
        super().__init__(**kwargs)
        self._trained = False
        self._num_classes = num_classes
        self._class_weight = class_weight
        self._is_multilabel = is_multilabel
        self._ctx = ctx
        self._segmenter = segmenter
        self._max_length = max_length
        self._embed_size = embed_size
        self._vocab = vocab
        self._embed_weight = embed_weight
        self._label2idx = label2idx

        self.meta = {
            'num_classes': self._num_classes,
            'class_weight': self._class_weight,
            'is_multilabel': self._is_multilabel,
            'label2idx': self._label2idx,
            'max_length': self._max_length,
            'segmenter': self._segmenter,
            'embed_size': self._embed_size}

    def _build(self):
        self._build_net()
        if self._is_multilabel:
            self._loss = mx.gluon.loss.SigmoidBCELoss()
        else:
            self._loss = mx.gluon.loss.SoftmaxCELoss(sparse_label=False)
        self._trainable = [self._net]
        self._initialize_net()

    def _get_or_build_dataset(self, dataset, X, y):
        assert (X and y) or dataset
        if dataset:
            return dataset
        return self._SimpleClassifyDataset(X, y, vocab=self._vocab,
                                           label2idx=self._label2idx,
                                           segmenter=self._segmenter,
                                           max_length=self._max_length)

    def _decode(self, x, threshold=0.5):
        return _decode(x, is_binary=self._num_classes == 2,
                       is_multilabel=self._is_multilabel,
                       threshold=threshold)

    def _debinarize(self, binarized_label):
        l = [[i for i, t in enumerate(bl) if t == 1] for bl in binarized_label]
        if self._is_multilabel:
            return l
        return list(itertools.chain(*l))

    def _decode_label(self, idx):
        if isinstance(idx[0], list):
            return [self.idx2labels(i) for i in idx]
        return self.idx2labels(idx)

    def _valid_log(self, valid_dataset):
        scores = self.score(dataset=valid_dataset)
        if self._is_multilabel or self._num_classes > 2:
            for l, p, r, f, _ in zip(self.idx2labels(
                    list(range(self._num_classes))), *scores):
                self.logger.info(f'label: {l} precision: {p}, '
                                 f'recall: {r}, f1: {f}')
        else:
            p, r, f, _ = scores
            self.logger.info(f'precision: {p}, recall: {r}, f1: {f}')

    def _calculate_loss(self, batch_inputs, batch_mask, batch_labels):
        return self._loss(
            self._net(batch_inputs.transpose(axes=(1, 0)),
                      batch_mask.transpose(axes=(1, 0))),
            batch_labels)

    @property
    def _batchify_fn(self):
        return gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1), Pad(axis=0),
            gluonnlp.data.batchify.Stack())

    def predict(self, X=None, dataset=None, batch_size=512,
                return_origin_label=True):
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

            logits = self._net(batch_inputs.transpose(axes=(1, 0)),
                               batch_mask.transpose(axes=(1, 0)))
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
            return precision_recall_fscore_support(y, predictions,
                                                   average='binary')
        else:
            return precision_recall_fscore_support(
                y, predictions, labels=list(range(self._num_classes)))

    def save_model(self, file_path: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'vocab.json'), 'w') as f:
                f.write(self._vocab.to_json())
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            for i, item in enumerate(self._trainable):
                item.export(os.path.join(temp_dir, f'hybrid-{0:02}'))
                item.save_parameters(os.path.join(temp_dir, f'{i:02}-params'))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def load_model(cls, file_path, ctx=mx.cpu()):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            with open(os.path.join(temp_dir, 'vocab.json')) as f:
                vocab = gluonnlp.Vocab.from_json(f.read())
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())

            meta['ctx'] = ctx
            meta['vocab'] = vocab
            clf = cls(**meta)
            for i, block in enumerate(clf._trainable):
                block.load_parameters(os.path.join(temp_dir, f'{i:02}-params'),
                                      ctx=ctx)
        clf._trained = True
        return clf


class TextCNNClassifier(DeepClassifier):

    def __init__(self, num_classes=2, class_weight=None,
                 is_multilabel=False, label2idx=None, ctx=mx.cpu(),
                 vocab=None, embed_weight=None, segmenter='jieba',
                 max_length=100, embed_size=100,
                 num_filters=(25, 50, 75, 100, 125, 150),
                 ngram_filter_sizes=(1, 2, 3, 4, 5, 6),
                 conv_layer_activation='tanh', dropout=0.5,
                 num_highway=1, **kwargs):
        super().__init__(num_classes=num_classes, class_weight=class_weight,
                         is_multilabel=is_multilabel, label2idx=label2idx,
                         ctx=ctx, vocab=vocab, embed_weight=embed_weight,
                         segmenter=segmenter, max_length=max_length,
                         embed_size=embed_size, **kwargs)
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

        if vocab is not None:
            self._build()

    def _build_net(self):
        self._net = _ClassifyBlockComposition(
            EmbeddingLayer(len(self._vocab),
                           self._embed_size, prefix='embed_'),
            TextCNN(embed_size=self._embed_size,
                    num_filters=self._num_filters,
                    ngram_filter_sizes=self._ngram_filter_sizes,
                    conv_layer_activation=self._conv_layer_activation,
                    output_size=self._num_classes,
                    dropout=self._dropout,
                    num_highway=self._num_highway,
                    prefix='encode_'))
        self.meta.update({'vocab_size': len(self._vocab)})

    @property
    def _batchify_fn(self):
        return gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1, min_length=max(self._ngram_filter_sizes)),
            Pad(axis=0, min_length=max(self._ngram_filter_sizes)),
            gluonnlp.data.batchify.Stack())


class TextRNNClassifier(DeepClassifier):

    def __init__(self, num_classes=2, class_weight=None,
                 is_multilabel=False, label2idx=None, ctx=mx.cpu(),
                 vocab=None, embed_weight=None, segmenter='jieba',
                 max_length=100, embed_size=100,
                 hidden_size=512, num_rnn_layers=1, output_size=1,
                 dropout=0.5, dense_connection='last', **kwargs):
        super().__init__(num_classes=num_classes, class_weight=class_weight,
                         is_multilabel=is_multilabel, label2idx=label2idx,
                         ctx=ctx, vocab=vocab, embed_weight=embed_weight,
                         segmenter=segmenter, max_length=max_length,
                         embed_size=embed_size, **kwargs)
        self._hidden_size = hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._dropout = dropout
        self._dense_connection = dense_connection

        self.meta.update({
            'hidden_size': hidden_size,
            'num_rnn_layers': num_rnn_layers,
            'dropout': dropout,
            'dense_connection': dense_connection})

        if vocab is not None:
            self._build()

    def _build_net(self):
        self._net = _ClassifyBlockComposition(
            EmbeddingLayer(len(self._vocab),
                           self._embed_size, prefix='embed_'),
            TextRNN(hidden_size=self._hidden_size,
                    num_rnn_layers=self._num_rnn_layers,
                    dropout=self._dropout,
                    dense_connection=self._dense_connection,
                    output_size=self._num_classes,
                    prefix='encode_'))
        self.meta.update({'vocab_size': len(self._vocab)})
