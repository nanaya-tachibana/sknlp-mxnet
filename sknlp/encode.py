import os
import tempfile
import shutil
import mxnet as mx
from mxnet.gluon import nn, rnn

import gluonnlp


class SequenceEncode(nn.HybridBlock):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hybrid_forward(self, F, inputs, mask=None):
        raise NotImplementedError('forward function is not implemented.')

    def save(self, file_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.export(os.path.join(temp_dir, 'encode'))
            shutil.make_archive(file_path, 'gztar', temp_dir)

    @classmethod
    def load(cls, file_path, prefix='encode_', use_mask=True, ctx=mx.cpu()):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'gztar')
            inputs = ['data0', 'data1'] if use_mask else ['data']
            ins = nn.SymbolBlock.imports(
                os.path.join(temp_dir, 'encode-symbol.json'), inputs,
                os.path.join(temp_dir, 'encode-0000.params'), ctx=ctx
            )
            ins.hybridize()
        return ins


class TextCNN(SequenceEncode):

    def __init__(
        self, embed_size=100, num_filters=(25, 50, 75, 100),
        ngram_filter_sizes=(1, 2, 3, 4), conv_layer_activation='tanh',
        output_size=1, dropout=0, num_highway=1, **kwargs
    ):
        super().__init__(**kwargs)
        with self.name_scope():
            self.cnn_layer = gluonnlp.model.ConvolutionalEncoder(
                embed_size=embed_size, num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                output_size=None, num_highway=num_highway, prefix='cnn_'
            )
            self.cnn_dropout = nn.Dropout(dropout, prefix='cnndropout_')
            self.dense_layer = nn.Dense(
                output_size, flatten=False, prefix='dense_'
            )

    def hybrid_forward(self, F, inputs, mask=None):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        return self.dense_layer(self.cnn_dropout(
            self.cnn_layer(inputs, mask)
        ))


class TextRNN(SequenceEncode):

    def __init__(
        self, hidden_size=512, num_rnn_layers=1, output_size=1,
        dropout=0.0, dense_connection='last', **kwargs
    ):
        super().__init__(**kwargs)
        self._dense_connection = dense_connection
        with self.name_scope():
            self.rnn_layer = rnn.LSTM(
                hidden_size // 2, num_rnn_layers, dropout=dropout,
                bidirectional=True, prefix='rnn_'
            )
            self.rnn_dropout = nn.Dropout(dropout, prefix='rnndropout_')
            self.dense_layer = nn.Dense(
                output_size, flatten=False, prefix='dense_'
            )

    def hybrid_forward(self, F, inputs, mask=None):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        rnn_output = self.rnn_layer(inputs)
        if self._dense_connection == 'last' and mask is not None:
            rnn_output = F.concat(
                F.SequenceLast(
                    rnn_output, sequence_length=F.sum(mask, axis=0),
                    use_sequence_length=True
                ),
                F.squeeze(
                    F.slice_axis(rnn_output, axis=0, begin=0, end=1), axis=0
                )
            )
        return self.dense_layer(self.rnn_dropout(rnn_output))


class TextRCNN(SequenceEncode):

    def __init__(
        self, embed_size=100, num_filters=(25, 50, 75, 100),
        ngram_filter_sizes=(1, 2, 3, 4), conv_layer_activation='tanh',
        num_highway=1, rnn_hidden_size=512, num_rnn_layers=1, output_size=1,
        dropout=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        with self.name_scope():
            self.rnn_layer = rnn.LSTM(
                rnn_hidden_size // 2, num_rnn_layers, dropout=dropout,
                bidirectional=True, prefix='rnn_'
            )
            self.rnn_dropout = nn.Dropout(dropout, prefix='rnndropout_')
            self.cnn_layer = gluonnlp.model.ConvolutionalEncoder(
                embed_size=rnn_hidden_size, num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                output_size=None, num_highway=num_highway, prefix='cnn_'
            )
            self.cnn_dropout = nn.Dropout(dropout, prefix='cnndropout_')
            self.dense_layer = nn.Dense(
                output_size, flatten=False, prefix='dense_'
            )

    def hybrid_forward(self, F, inputs, mask=None):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        return self.dense_layer(self.cnn_dropout(self.cnn_layer(
            self.rnn_dropout(self.rnn_layer(inputs)), mask
        )))
