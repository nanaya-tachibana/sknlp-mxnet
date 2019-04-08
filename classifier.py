import numpy as np
from scipy.special import expit
import mxnet as mx
from mxnet.gluon import nn, rnn

import gluonnlp

from base import DeepModelTrainMixin
from dataset import _SimpleClassifyDataset
from utils import word_cut_func, Pad


class EmbeddingLayer(nn.HybridBlock):

    def __init__(self, vocab_size, embed_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(vocab_size, embed_size),
                init=mx.init.Uniform(0.1), grad_stype='row_sparse')

    def hybrid_forward(self, F, inputs, mask, weight):
        """
        inputs: shape(seq_length, batch_size)
        """
        return F.Embedding(inputs, weight,
                           self._vocab_size, self._embed_size,
                           sparse_grad=True)


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
            self.birnn_layer = rnn.LSTM(hidden_size // 2, num_rnn_layers,
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
        rnn_output = self.rnn_layer(inputs, mask)
        if self._dense_connection == 'last':
            rnn_output = F.squeeze(
                F.slice_axis(rnn_output, axis=0, begin=-1, end=None), axis=0)
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


class _SequenceBlock(nn.HybridSequential):

    def hybrid_forward(self, F, inputs, mask):
        x = inputs
        for block in self._children.values():
            x = block(x, mask)
        return x


class TextCNNClassifier(DeepModelTrainMixin):

    def __init__(self, num_classes, is_multilabel=False,
                 ctx=mx.cpu(), vocab=None, embed_weight=None,
                 segmenter=word_cut_func, max_length=100, embed_size=100,
                 num_filters=(25, 50, 75, 100),
                 ngram_filter_sizes=(1, 2, 3, 4),
                 conv_layer_activation='tanh',
                 dropout=0.5, num_highway=1):
        super().__init__()
        self._trained = False
        self._num_classes = num_classes
        self._is_multilabel = is_multilabel
        self._ctx = ctx
        self._segmenter = segmenter
        self._max_length = max_length

        self._embed_size = embed_size
        self._vocab = vocab
        self._embed_weight = embed_weight
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._conv_layer_activation = conv_layer_activation
        self._dropout = dropout
        self._num_highway = num_highway

        if vocab is not None:
            self._build_net()

    def _build_net(self):

        self._net = _SequenceBlock()
        with self._net.name_scope():
            self._net.add(EmbeddingLayer(len(self._vocab),
                                         self._embed_size,
                                         prefix='embed_'))
            self._net.add(TextCNN(
                embed_size=self._embed_size,
                num_filters=self._num_filters,
                ngram_filter_sizes=self._ngram_filter_sizes,
                conv_layer_activation=self._conv_layer_activation,
                output_size=self._num_classes,
                dropout=self._dropout,
                num_highway=self._num_highway,
                prefix='encode_'))
        if self._is_multilabel:
            self._loss = mx.gluon.loss.SigmoidBCELoss()
        else:
            self._loss = mx.gluon.loss.SoftmaxCELoss(sparse_label=False)

        self._trainable = [self._net]
        self.meta = {
            'vocab_size': len(self._vocab),
            'embed_size': self._embed_size,
            'max_length': self._max_length,
            'num_filters': self._num_filters,
            'ngram_filter_sizes': self._ngram_filter_sizes,
            'conv_layer_activation': self._conv_layer_activation,
            'output_size': self._num_classes,
            'dropout': self._dropout,
            'num_highway': self._num_highway}

        self._net.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        self._loss.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        if self._embed_weight is not None:
            self._embed_layer.weight.set_data(self._embed_weight)
        # self._net.hybridize()
        # self._loss.hybridize()

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
        assert (X and y) or train_dataset

        if not update_embedding:
            self.net[0].weight.grad_req = 'null'

        if train_dataset is None:
            train_dataset = _SimpleClassifyDataset(X, y, vocab=self._vocab,
                                                   segmenter=self._segmenter,
                                                   max_length=self._max_length)
        assert self._num_classes == len(train_dataset.label2idx)

        self.idx2labels = train_dataset.idx2labels
        if self._vocab is None:
            self._vocab = train_dataset.vocab
            self._build_net()
        else:
            train_dataset.vocab = self._vocab

        if valid_X and valid_y:
            valid_dataset = _SimpleClassifyDataset(
                valid_X, valid_y, vocab=self._vocab,
                label2idx=train_dataset.label2idx, segmenter=self._segmenter,
                max_length=self._max_length)

        train_batchify_fn = gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1), Pad(axis=0),
            gluonnlp.data.batchify.Stack())

        train_dataloader = mx.gluon.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            last_batch=last_batch,
            batchify_fn=train_batchify_fn)

        self._fit(train_dataloader, lr, n_epochs,
                  valid_dataset=valid_dataset,
                  optimizer=optimizer, clip=clip, verbose=verbose,
                  checkpoint=checkpoint, save_frequency=save_frequency)
        self._trained = True

    def _decode(self, x, threshold=0.5):
        res = _decode(x, is_binary=self._num_classes == 2,
                      is_multilabel=self._is_multilabel, threshold=threshold)
        if self._is_multilabel:
            return [
                [self.idx2labels([i])[0] for i, t in enumerate(r) if t == 1]
                for r in res]
        else:
            return self.idx2labels(res)

    def _valid_log(self, valid_dataset):
        scores = self.valid_score(valid_dataset[0],
                                  valid_dataset[1],
                                  valid_dataset[2])
        self.logger.info(
            'province(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'city(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'district(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'location(ser: %.3f, cer: %.3f)\n' % tuple(scores))

    def _calculate_loss(self, batch_inputs, batch_mask, batch_labels):
        return self._loss(
            self._net(batch_inputs.transpose(axes=(1, 0)),
                      batch_mask.transpose(axes=(1, 0))),
            batch_labels)

    def predict(self, X, batch_size=512):
        dataset = _SimpleClassifyDataset(X, ['O'] * len(X),
                                         vocab=self._vocab,
                                         segmenter=self._segmenter,
                                         max_length=self._max_length)

        batchify_fn = gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1), Pad(axis=0),
            gluonnlp.data.batchify.Stack())

        dataloader = mx.gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            batchify_fn=batchify_fn)

        predictions = []
        for (batch_inputs, batch_mask, _) in dataloader:
            batch_inputs = batch_inputs.as_in_context(self._ctx)
            batch_mask = batch_mask.as_in_context(self._ctx)

            logits = self._net(batch_inputs.transpose(axes=(1, 0)),
                               batch_mask.transpose(axes=(1, 0)))
            predictions.extend(self._decode(logits.asnumpy()))
        return predictions
