import json
import itertools

import numpy as np
from scipy.special import expit
import mxnet as mx
from mxnet.gluon import nn, rnn

import gluonnlp
from sklearn.metrics import precision_recall_fscore_support

from .base import DeepModelTrainMixin
from .dataset import _SimpleClassifyDataset
from .utils import word_cut_func, Pad
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

    def hybrid_forward(self, F, inputs, mask):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        rnn_output = self.rnn_layer(inputs)
        if self._dense_connection == 'last':
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

    def hybrid_forward(self, F, inputs, mask):
        return self.encode_layer(self.embedding_layer(inputs), mask)


class DeepClassifier(DeepModelTrainMixin):

    def __init__(self, num_classes, class_weight=None,
                 is_multilabel=False, ctx=mx.cpu(),
                 vocab=None, embed_weight=None,
                 segmenter=word_cut_func, max_length=100, embed_size=100):
        super().__init__()
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

        self._label2idx = None

    def _build_net(self):
        """
        Implement this function to build net.
        """
        raise NotImplementedError('build net is not implemented.')

    def _build(self):
        self._build_net()
        if self._is_multilabel:
            self._loss = mx.gluon.loss.SigmoidBCELoss()
        else:
            self._loss = mx.gluon.loss.SoftmaxCELoss(sparse_label=False)
        self._trainable = [self._net]
        self._initialize_net()

    def _initialize_net(self):
        self._net.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        self._loss.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        if self._embed_weight is not None:
            self._net.embedding_layer.weight.set_data(self._embed_weight)
        self._net.hybridize()
        self._loss.hybridize()

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

    def _build_dataloader(self, dataset, batch_size, shuffle, last_batch):
        return mx.gluon.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        last_batch=last_batch,
                                        batchify_fn=self._batchify_fn)

    def _get_or_build_dataset(self, dataset, X, y):
        assert (X and y) or dataset
        if dataset:
            return dataset
        return _SimpleClassifyDataset(X, y, vocab=self._vocab,
                                      label2idx=self._label2idx,
                                      segmenter=self._segmenter,
                                      max_length=self._max_length)

    def fit(self, X=None, y=None, train_dataset=None,
            valid_X=None, valid_y=None, valid_dataset=None,
            batch_size=32, last_batch='keep',
            update_embedding=True, n_epochs=15,
            optimizer='adam', lr=3e-4, clip=5.0, verbose=True,
            checkpoint=None, save_frequency=1):
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
        train_dataset = self._get_or_build_dataset(train_dataset, X, y)
        print(train_dataset.label2idx)
        assert self._num_classes == len(train_dataset.label2idx)

        self.idx2labels = train_dataset.idx2labels
        if self._vocab is None:
            self._vocab = train_dataset.vocab
            self._build()
        if self._label2idx is None:
            self._label2idx = train_dataset.label2idx

        if valid_X and valid_y and valid_dataset is None:
            valid_dataset = self._get_or_build_dataset(train_dataset, X, y)

        if not update_embedding:
            self._net.embedding_layer.weight.grad_req = 'null'

        dataloader = self._build_dataloader(train_dataset, batch_size,
                                            True, last_batch)
        self._fit(dataloader, lr, n_epochs,
                  valid_dataset=valid_dataset,
                  optimizer=optimizer, clip=clip, verbose=verbose,
                  checkpoint=checkpoint, save_frequency=save_frequency)

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


class TextCNNClassifier(DeepClassifier):

    def __init__(self, num_classes, is_multilabel=False,
                 ctx=mx.cpu(), vocab=None, embed_weight=None,
                 segmenter=word_cut_func, max_length=100, embed_size=100,
                 num_filters=(25, 50, 75, 100),
                 ngram_filter_sizes=(1, 2, 3, 4),
                 conv_layer_activation='tanh', dropout=0.5, num_highway=1):
        super().__init__(num_classes, is_multilabel=is_multilabel, ctx=ctx,
                         vocab=vocab, embed_weight=embed_weight,
                         segmenter=segmenter, max_length=max_length,
                         embed_size=embed_size)
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._conv_layer_activation = conv_layer_activation
        self._dropout = dropout
        self._num_highway = num_highway

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
        self.meta = {
            'num_classes': self._num_classes,
            'is_multilabel': self._is_multilabel,
            'label2idx': self._label2idx,
            'vocab_size': len(self._vocab),
            'embed_size': self._embed_size,
            'max_length': self._max_length,
            'num_filters': list(self._num_filters),
            'ngram_filter_sizes': list(self._ngram_filter_sizes),
            'conv_layer_activation': self._conv_layer_activation,
            'dropout': self._dropout,
            'num_highway': self._num_highway}

    @property
    def _batchify_fn(self):
        return gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1, min_length=max(self._ngram_filter_sizes)),
            Pad(axis=0, min_length=max(self._ngram_filter_sizes)),
            gluonnlp.data.batchify.Stack())

    @classmethod
    def load_model(cls, vocab_file, meta_file, model_file,
                   segmenter=word_cut_func, ctx=mx.cpu()):
        with open(vocab_file) as f:
            vocab = gluonnlp.Vocab.from_json(f.read())
        with open(meta_file) as f:
            meta = json.loads(f.read())

        clf = cls(meta['num_classes'],
                  is_multilabel=meta['is_multilabel'],
                  ctx=mx.cpu(),
                  vocab=vocab,
                  segmenter=segmenter,
                  max_length=meta['max_length'],
                  embed_size=meta['embed_size'],
                  num_filters=meta['num_filters'],
                  ngram_filter_sizes=meta['ngram_filter_sizes'],
                  conv_layer_activation=meta['conv_layer_activation'],
                  dropout=meta['dropout'],
                  num_highway=meta['num_highway'])
        clf._label2idx = meta['label2idx']
        for filename, block in zip(model_file, clf._trainable):
            block.load_parameters(filename, ctx=ctx)
        clf._trained = True
        return clf


class TextRNNClassifier(DeepClassifier):

    def __init__(self, num_classes, is_multilabel=False,
                 ctx=mx.cpu(), vocab=None, embed_weight=None,
                 segmenter=word_cut_func, max_length=100, embed_size=100,
                 hidden_size=512, num_rnn_layers=1, output_size=1,
                 dropout=0.5, dense_connection='last'):
        super().__init__(num_classes, is_multilabel=is_multilabel, ctx=ctx,
                         vocab=vocab, embed_weight=embed_weight,
                         segmenter=segmenter, max_length=max_length,
                         embed_size=embed_size)
        self._hidden_size = hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._dropout = dropout
        self._dense_connection = dense_connection

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
        self.meta = {
            'num_classes': self._num_classes,
            'is_multilabel': self._is_multilabel,
            'max_length': self._max_length,
            'vocab_size': len(self._vocab),
            'embed_size': self._embed_size,
            'hidden_size': self._hidden_size,
            'num_rnn_layers': self._num_rnn_layers,
            'dropout': self._dropout,
            'dense_connection': self._dense_connection}

    @classmethod
    def load_model(cls, vocab_file, meta_file, model_file,
                   segmenter=word_cut_func, ctx=mx.cpu()):
        with open(vocab_file) as f:
            vocab = gluonnlp.Vocab.from_json(f.read())
        with open(meta_file) as f:
            meta = json.loads(f.read())
        clf = cls(meta['num_classes'],
                  is_multilabel=meta['is_multilabel'],
                  ctx=mx.cpu(),
                  vocab=vocab,
                  segmenter=segmenter,
                  max_length=meta['max_length'],
                  embed_size=meta['embed_size'],
                  hidden_size=meta['hidden_size'],
                  num_rnn_layers=meta['num_rnn_layers'],
                  output_size=meta['num_classes'],
                  dropout=meta['dropout'],
                  dense_connection=meta['dense_connection'])
        for filename, block in zip(model_file, clf._trainable):
            block.load_parameters(filename, ctx=ctx)
        return clf
