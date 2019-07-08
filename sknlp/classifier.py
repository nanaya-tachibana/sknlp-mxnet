import itertools
import functools
import json
import os
import shutil
import tempfile
import logging

import mxnet as mx
from mxnet.gluon import nn

import gluonnlp

import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_fscore_support

from .base import DeepSupervisedModel
from .data import ClassifyDataset, InMemoryDataset
from .data.batchify import Pad, Stack
from .utils.array import sequence_mask
from .embedding import Token2vec
from .encode import TextCNN, TextRCNN, TextRNN, TextTransformer
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


def batchify(padding, one_batch):
    (inputs, length), labels = gluonnlp.data.batchify.Tuple(
        Pad(axis=0, pad_val=padding, ret_length=True), Stack()
    )(one_batch)
    inputs = inputs.transpose((1, 0))
    mask = sequence_mask(np.ones_like(inputs), length.astype('int'))
    return inputs, mask, labels.astype('float32')


class DeepClassifier(DeepSupervisedModel):

    def __init__(
        self, num_classes, encode_layer, embedding_layer=None,
        vocab=None, is_multilabel=False, label2idx=None,
        segmenter='jieba', max_length=100, embed_size=100,
        threshold=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.encode_layer = encode_layer
        self._vocab = vocab
        self._num_classes = num_classes
        self._is_multilabel = is_multilabel
        self._segmenter = segmenter
        self._cut = Segmenter(segmenter).cut
        self._max_length = max_length
        self._embed_size = embed_size
        self._label2idx = label2idx
        self._threshold = threshold or dict()

        self.meta = {
            'num_classes': num_classes,
            'is_multilabel': is_multilabel,
            'label2idx': label2idx,
            'max_length': max_length,
            'segmenter': segmenter,
            'embed_size': embed_size,
            'threshold': threshold
        }

    def _build(self, ctx, initialize=True):
        if self.embedding_layer is None:
            self.embedding_layer = Token2vec(
                self._vocab, self._embed_size, loss=None
            )
        self._trainable = {
            'embedding': self.embedding_layer,
            'encode': self.encode_layer
        }
        if self._is_multilabel:
            self.loss = mx.gluon.loss.SigmoidBCELoss()
        else:
            self.loss = mx.gluon.loss.SoftmaxCELoss(sparse_label=False)
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
        return ClassifyDataset(
            d, vocab=self._vocab, label2idx=self._label2idx,
            segmenter=self._cut, max_length=self._max_length
        )

    def _decode(self, x, threshold):
        threshold = np.array([
            threshold.get(l, 0.5)
            for l in self.idx2labels(range(self._num_classes))
        ])
        return _decode(
            x, threshold, is_binary=self._num_classes == 2,
            is_multilabel=self._is_multilabel
        )

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
            return f
        else:
            p, r, f, _ = scores
            logger.info(f'precision: {p}, recall: {r}, f1: {f}')
            return f

    def _calculate_logits(self, inputs, mask, *args):
        return self.encode_layer(self.embedding_layer(inputs), mask)

    def _calculate_loss(self, inputs, mask, labels):
        logits = self._calculate_logits(inputs, mask)
        return self.loss(logits, labels), None

    def _batchify_fn(self):
        vocab = self._vocab
        return functools.partial(batchify, vocab[vocab.padding_token])

    def predict(
        self, X=None, dataset=None, batch_size=512,
        threshold=None, return_origin_label=True
    ):
        assert self._trained
        assert dataset or X
        if threshold is None:
            threshold = self._threshold

        if dataset is None:
            dataset = self._get_or_build_dataset(dataset, X, ['O'] * len(X))
        self.idx2labels = dataset.idx2labels
        dataloader = self._build_dataloader(dataset, batch_size, False, 'keep')

        predictions = []
        for one_batch in dataloader:
            for logits in self._forward(
                self._calculate_logits, one_batch, self._ctx, batch_axis=1
            ):
                predictions.extend(self._decode(logits.asnumpy(), threshold))
        dataloader.reset()
        if return_origin_label:
            if self._is_multilabel:
                predictions = self._debinarize(predictions)
            return self._decode_label(predictions)
        return predictions

    def score(
        self, X=None, y=None, dataset=None, batch_size=512
    ):
        assert self._trained
        dataset = self._get_or_build_dataset(dataset, X, y)
        predictions = self.predict(
            dataset=dataset,
            batch_size=batch_size,
            return_origin_label=False
        )
        y = [label for _, label in dataset]
        if not self._is_multilabel:
            y = [np.argmax(label) for label in y]
        y = np.vstack(y)
        predictions = np.vstack(predictions)
        if self._num_classes == 2 and not self._is_multilabel:
            binary_score = precision_recall_fscore_support(
                y, predictions, average='binary'
            )
            return binary_score
        else:
            detail_score = precision_recall_fscore_support(
                y, predictions, labels=list(range(self._num_classes))
            )
            micro_score = precision_recall_fscore_support(
                y, predictions,
                labels=list(range(self._num_classes)), average='micro'
            )
            return detail_score, micro_score

    def save(self, file_path: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.embedding_layer.save(os.path.join(temp_dir, 'embedding'))
            self.encode_layer.export(os.path.join(temp_dir, 'encode'))
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def _load_embedding_layer(cls, file_path, update, ctx):
        return Token2vec.load(file_path, update=update, ctx=ctx)

    @classmethod
    def _load(cls, temp_dir, meta, inputs=None, update=False, ctx=mx.cpu()):
        embedding_layer = cls._load_embedding_layer(
            os.path.join(temp_dir, 'embedding.tar.gz'), update, ctx
        )
        if inputs is None:
            inputs = ['data0', 'data1']
        encode_layer = nn.SymbolBlock.imports(
            os.path.join(temp_dir, 'encode-symbol.json'), inputs,
            os.path.join(temp_dir, 'encode-0000.params'), ctx=ctx
        )
        for name, param in encode_layer.collect_params().items():
            param.grad_req = 'null'
        ins = cls(
            meta['num_classes'], encode_layer,
            embedding_layer=embedding_layer,
            vocab=embedding_layer._vocab,
            is_multilabel=meta['is_multilabel'],
            label2idx=meta['label2idx'], segmenter=meta['segmenter'],
            max_length=meta['max_length'], embed_size=meta['embed_size'],
            threshold=meta['threshold'], ctx=ctx
        )
        ins._trained = True
        ins._build(ctx, initialize=False)
        return ins

    @staticmethod
    def load(file_path, update=False, ctx=mx.cpu()):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())

            if meta['model_type'] == 'builtin-text_transformer_classifier':
                return TextTransformerClassifier._load(
                    temp_dir, meta, update=update,
                    inputs=['data0', 'data1', 'data2'], ctx=ctx
                )
            elif meta['model_type'] == 'builtin-text_cnn_classifier':
                return TextCNNClassifier._load(
                    temp_dir, meta, update=update, ctx=ctx
                )
            elif meta['model_type'] == 'builtin-text_rnn_classifier':
                return TextRNNClassifier._load(
                    temp_dir, meta, update=update, ctx=ctx
                )
            elif meta['model_type'] == 'builtin-text_rcnn_classifier':
                return TextRCNNClassifier._load(
                    temp_dir, meta, update=update, ctx=ctx
                )
            else:
                raise ValueError('unknown model type.')


def cnn_batchify(padding, min_length, one_batch):
    (inputs, length), labels = gluonnlp.data.batchify.Tuple(
        Pad(axis=0, pad_val=padding, ret_length=True, min_length=min_length),
        Stack()
    )(one_batch)
    inputs = inputs.transpose((1, 0))
    mask = sequence_mask(np.ones_like(inputs), length.astype('int'))
    return inputs, mask, labels.astype('float32')


class TextCNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        vocab=None, is_multilabel=False, label2idx=None, segmenter='jieba',
        max_length=100, embed_size=100, threshold=None,
        num_filters=(25, 50, 75, 100), ngram_filter_sizes=(1, 2, 3, 4),
        conv_layer_activation='tanh', num_highway=1, dropout=0,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='tanh',
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextCNN(
                embed_size=embed_size,
                num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                num_highway=num_highway,
                dropout=dropout,
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
            threshold=threshold, ctx=ctx, **kwargs
        )
        self.meta.update({
            'num_filters': list(num_filters),
            'ngram_filter_sizes': list(ngram_filter_sizes),
            'conv_layer_activation': conv_layer_activation,
            'num_highway': num_highway,
            'num_fc_layers': num_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'fc_activation': fc_activation
        })
        self.meta.update({'model_type': 'builtin-text_cnn_classifier'})

    def _batchify_fn(self):
        vocab = self._vocab
        return functools.partial(
            cnn_batchify, vocab[vocab.padding_token],
            min(self.meta['ngram_filter_sizes'])
        )


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
            threshold=threshold, ctx=ctx, **kwargs
        )
        self.meta.update({
            'num_rnn_layers': num_rnn_layers,
            'projection_size': projection_size,
            'hidden_size': hidden_size,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip,
            'dropout': dropout,
            'dense_connection': dense_connection,
            'num_fc_layers': num_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'fc_activation': fc_activation
        })
        self.meta.update({'model_type': 'builtin-text_rnn_classifier'})


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
            threshold=threshold, ctx=ctx, **kwargs
        )
        self.meta.update({
            'num_rnn_layers': num_rnn_layers,
            'projection_size': projection_size,
            'hidden_size': hidden_size,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip,
            'kmax': kmax,
            'dropout': dropout,
            'num_fc_layers': num_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'fc_activation': fc_activation
        })
        self.meta.update({'model_type': 'builtin-text_rcnn_classifier'})


class TextTransformerClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100, threshold=None,
        attention_cell='multi_head', num_layers=2,
        units=512, hidden_size=2048,
        num_heads=4, scaled=True, dropout=0.0,
        use_residual=True, output_attention=False,
        weight_initializer=None, bias_initializer='zeros',
        positional_weight='learned', use_bert_encoder=True,
        use_layer_norm_before_dropout=False, scale_embed=True,
        prefix=None, params=None, output_size=1,
        num_fc_layers=1, fc_hidden_size=512, fc_activation='relu',
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextTransformer(
                attention_cell=attention_cell,
                num_layers=num_layers,
                units=units,
                hidden_size=hidden_size,
                max_length=max_length,
                num_heads=num_heads,
                scaled=scaled,
                dropout=dropout,
                use_residual=use_residual,
                output_attention=output_attention,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                params=params,
                # extra configurations for transformer
                positional_weight=positional_weight,
                use_bert_encoder=use_bert_encoder,
                use_layer_norm_before_dropout=use_layer_norm_before_dropout,
                scale_embed=scale_embed,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
                fc_activation=fc_activation,
                output_size=num_classes,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter,
            max_length=max_length, embed_size=embed_size,
            threshold=threshold, ctx=ctx, **kwargs
        )
        self.meta.update({
            'attention_cell': attention_cell,
            'num_layers': num_layers,
            'units': units,
            'hidden_size': hidden_size,
            'max_length': max_length,
            'num_heads': num_heads,
            'scaled': scaled,
            'dropout': dropout,
            'use_residual': use_residual,
            'output_attention': output_attention,
            'weight_initializer': weight_initializer,
            'bias_initializer': bias_initializer,
        })
        self.meta.update({'model_type': 'builtin-text_transformer_classifier'})
