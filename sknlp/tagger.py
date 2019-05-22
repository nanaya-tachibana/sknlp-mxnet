import itertools

import os
import tempfile
import shutil
import json
import logging

import mxnet as mx
from sklearn.metrics import precision_recall_fscore_support

import gluonnlp

from .base import DeepSupervisedModel
from .data import Pad, InMemoryDataset, SequenceTagDataset
from .embedding import NonContextEmbeddingLayer
from .crf import Crf, viterbi_decode
from .encode import TextRNN

logger = logging.getLogger(__name__)


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

    def _build(self):
        if self.embedding_layer is None:
            self.embedding_layer = NonContextEmbeddingLayer(
                self._vocab, self._embed_size, prefix='embed_'
            )
        self.meta['embedding_prefix'] = self.embedding_layer.prefix
        self.meta['encode_prefix'] = self.encode_layer.prefix
        self._loss = Crf(self._num_classes, prefix='crf_')
        self.meta['crf_prefix'] = self._loss.prefix
        self._trainable = {
            'embedding': self.embedding_layer,
            'encode': self.encode_layer,
            'loss': self._loss
        }

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
            self._loss.transitions.data().asnumpy())
        scores = self.score(dataset=valid_dataset)
        for l, p, r, f, _ in zip(self.idx2labels(
                list(range(self._num_classes))), *scores):
            logger.info(
                f'label: {l} precision: {p}, recall: {r}, f1: {f}'
            )

    def _calculate_loss(self, batch_inputs, batch_mask, batch_labels):
        return -self._loss(
            self.encode_layer(
                self.embedding_layer(batch_inputs.transpose(axes=(1, 0)))
            ),
            batch_labels.transpose(axes=(1, 0)),
            batch_mask.transpose(axes=(1, 0))
        )

    def _create_decoder(self, transitions):

        def decoder(inputs, mask=None):
            return viterbi_decode(transitions, inputs, mask=mask)

        return decoder

    def _batchify_fn(self):

        def batchify(one_batch):
            (batch_inputs, batch_length), batch_labels = \
                gluonnlp.data.batchify.Tuple(
                    Pad(axis=0, pad_val=self._vocab['<pad>'], ret_length=True),
                    Pad(axis=0), Pad(axis=0, pad_val=self._label2idx['O'])
            )(one_batch)
            batch_inputs = batch_inputs.transpose(axes=(1, 0))
            batch_mask = mx.nd.SequenceMask(
                mx.nd.ones_like(batch_inputs),
                sequence_length=batch_length.astype('float32'),
                use_sequence_length=True
            )
            batch_labels = batch_labels.transpose(axes=(1, 0))
            return (
                batch_inputs.as_in_context(self._ctx),
                batch_mask.as_in_context(self._ctx),
                batch_labels.as_in_context(self._ctx)
            )

        return batchify

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
            logits = self.encode_layer(self.embedding_layer(
                batch_inputs.transpose(axes=(1, 0))
            ))
            predictions.extend(self._decode(
                logits.asnumpy(),
                batch_mask.transpose(axes=(1, 0)).asnumpy()
            ))
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

    def save(self, file_path: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.embedding_layer.save(os.path.join(temp_dir, 'embedding'))
            self.encode_layer.save(os.path.join(temp_dir, 'encode'))
            self._loss.save(os.path.join(temp_dir, 'crf_loss'))
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def _load_embedding_layer(cls, file_path, prefix, update, ctx):
        return NonContextEmbeddingLayer.load(
            file_path, prefix=prefix, update=update, ctx=ctx
        )

    @classmethod
    def _load_encode_layer(cls, file_path, prefix):
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
            loss = Crf.load(
                os.path.join(temp_dir, 'crf_loss.tar.gz'),
                prefix=meta.pop('crf_prefix')
            )

            meta['ctx'] = ctx
            meta['vocab'] = embedding_layer._vocab
            meta['embedding_layer'] = embedding_layer
            clf = cls(encode_layer, **meta)
            clf.encode_layer = encode_layer
        clf._decode = clf._create_decoder(
            loss.params.get('transitions').data().asnumpy()
        )
        clf._trained = True
        return clf


class TextRNNTagger(DeepTagger):

    def __init__(
        self, num_tags, embedding_layer=None, label2idx=None, vocab=None,
        segmenter=None, max_length=100, embed_size=100, hidden_size=512,
        num_rnn_layers=1, output_size=1, dropout=0.5, dense_connection=None,
        ctx=mx.cpu(), **kwargs
    ):
        encode_layer = TextRNN(
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            dropout=dropout,
            dense_connection=dense_connection,
            output_size=num_tags,
            prefix='encode_'
        )
        super().__init__(
            num_tags, encode_layer, embedding_layer=embedding_layer,
            label2idx=label2idx, vocab=vocab, segmenter=segmenter,
            max_length=max_length, embed_size=embed_size, ctx=ctx, **kwargs)
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
        return TextRNN.load(file_path, use_mask=False, prefix=prefix, ctx=ctx)
