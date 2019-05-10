import itertools

import os
import tempfile
import shutil
import json

import mxnet as mx
from mxnet.gluon import nn
from sklearn.metrics import precision_recall_fscore_support

import gluonnlp

from .base import DeepModel
from .data import Pad
from .data.data import _SimpleSequenceTagDataset
from .embedding import NonContextEmbeddingLayer
from .crf import Crf, viterbi_decode
from .classifier import TextRNN
from .classifier import _ClassifyBlockComposition as _TagBlockComposition


class DeepTagger(DeepModel):

    def __init__(self, num_tags, label2idx=None,
                 vocab=None, embed_weight=None, segmenter=None,
                 max_length=100, embed_size=100, embedding_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._trained = False
        self._num_classes = num_tags
        self._label2idx = label2idx
        self._segmenter = segmenter
        self._max_length = max_length
        self._embed_size = embed_size
        self._vocab = vocab

        if embedding_layer is None:
            self.embedding_layer = NonContextEmbeddingLayer(
                len(vocab), embed_size, prefix='embed_'
            )
        else:
            self.embedding_layer = embedding_layer
            self._embed_size = embedding_layer._embed_size

        self._embed_weight = embed_weight

        self.meta = {
            'num_tags': self._num_classes,
            'label2idx': self._label2idx,
            'max_length': self._max_length,
            'segmenter': self._segmenter,
            'embed_size': self._embed_size}

    def _build(self):
        self._build_net()
        self._loss = Crf(self._num_classes)
        self._trainable = [self._net, self._loss]
        self._initialize_net()

    def _get_or_build_dataset(self, dataset, X, y):
        assert (X is not None and y is not None) or dataset is not None
        if dataset:
            return dataset
        return _SimpleSequenceTagDataset(X, y, vocab=self._vocab,
                                         label2idx=self._label2idx,
                                         segmenter=self._segmenter,
                                         max_length=self._max_length)

    def _valid_log(self, valid_dataset):
        self._decode = self._create_decoder(
            self._loss.transitions.data().asnumpy())
        scores = self.score(dataset=valid_dataset)
        for l, p, r, f, _ in zip(self.idx2labels(
                list(range(self._num_classes))), *scores):
            self.logger.info(f'label: {l} precision: {p}, '
                             f'recall: {r}, f1: {f}')

    def _calculate_loss(self, batch_inputs, batch_mask, batch_labels):
        return -self._loss(
            self._net(batch_inputs.transpose(axes=(1, 0))),
            batch_labels.transpose(axes=(1, 0)),
            batch_mask.transpose(axes=(1, 0)))

    def _create_decoder(self, transitions):

        def decoder(inputs, mask=None):
            return viterbi_decode(transitions, inputs, mask=mask)

        return decoder

    @property
    def _batchify_fn(self):
        return gluonnlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1), Pad(axis=0),
            Pad(axis=0, pad_val=self._label2idx['O']))

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
            logits = self._net(batch_inputs.transpose(axes=(1, 0)))
            predictions.extend(
                self._decode(logits.asnumpy(),
                             batch_mask.transpose(axes=(1, 0)).asnumpy()))
        if return_origin_label:
            return [self.idx2labels(idx) for idx in predictions]
        return predictions

    def score(self, X=None, y=None, dataset=None, batch_size=512):
        assert self._trained
        dataset = self._get_or_build_dataset(dataset, X, y)
        predictions = self.predict(dataset=dataset, return_origin_label=False)
        y = list(itertools.chain(*[label for _, _, label in dataset]))
        predictions = list(itertools.chain(*predictions))
        return precision_recall_fscore_support(
            y, predictions, labels=list(range(self._num_classes)))

    def save_model(self, file_path: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'vocab.json'), 'w') as f:
                f.write(self._vocab.to_json())
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            for i, item in enumerate(self._trainable):
                item.export(os.path.join(temp_dir, f'hybrid-{i:02}'))
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
        clf._decode = clf._create_decoder(
            clf._loss.transitions.data().asnumpy())
        clf._trained = True
        return clf


class TextRNNTagger(DeepTagger):

    def __init__(self, num_tags, label2idx=None, vocab=None, segmenter=None,
                 max_length=100, embed_size=100, hidden_size=512,
                 num_rnn_layers=1, output_size=1, dropout=0.5,
                 dense_connection=None, embedding_layer=None,
                 ctx=mx.cpu(), **kwargs):
        super().__init__(num_tags=num_tags, label2idx=label2idx, vocab=vocab,
                         segmenter=segmenter, max_length=max_length,
                         embed_size=embed_size,
                         embedding_layer=embedding_layer, ctx=ctx, **kwargs)
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
        self._net = _TagBlockComposition(
            self.embedding_layer,
            TextRNN(hidden_size=self._hidden_size,
                    num_rnn_layers=self._num_rnn_layers,
                    dropout=self._dropout,
                    dense_connection=self._dense_connection,
                    output_size=self._num_classes,
                    prefix='encode_'))
        self.meta.update({'vocab_size': len(self._vocab)})
