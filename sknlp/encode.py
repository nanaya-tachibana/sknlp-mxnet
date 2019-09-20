from mxnet.gluon import nn, contrib
from gluonnlp.model import ConvolutionalEncoder
from .module import BiLSTM, ConvEncoder


class _HybridConcurrent(contrib.nn.HybridConcurrent):

    def hybrid_forward(self, F, x, m):
        out = []
        for block in self._children.values():
            out.append(block(x, m))
        out = F.stack(*out, axis=self.axis)
        return out


class FCLayer(nn.HybridBlock):

    def __init__(
        self, num_layers, hidden_size=512, activation='relu', output_size=1,
        prefix='fc_', **kwargs
    ):
        super().__init__(**kwargs)
        self.net = nn.HybridSequential(prefix=prefix)
        with self.net.name_scope():
            for i in range(num_layers):
                if i == num_layers - 1:
                    self.net.add(nn.Dense(
                        output_size, flatten=False, prefix=f'layer{i}_'
                    ))
                else:
                    self.net.add(nn.Dense(
                        hidden_size, flatten=False, prefix=f'layer{i}_'
                    ))
                    if i != num_layers - 2:
                        self.net.add(nn.Activation('relu'))
                    else:
                        self.net.add(nn.Activation(activation))
                    self.net.add(nn.BatchNorm(axis=-1))

    def hybrid_forward(self, F, input):
        return self.net(input)


class AttentionCell(nn.HybridBlock):

    def __init__(self, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(1, activation=activation, flatten=False)

    def hybrid_forward(self, F, input, mask):
        """
        input: shape(seq_length, batch_size, dim)
        mask: shape(seq_length, batch_size)
        """
        logits = self.dense(input)
        mask = F.expand_dims(mask, axis=-1)
        neg = -1e18 * F.ones_like(logits)
        scores = F.softmax(F.where(mask, logits, neg), axis=0) * mask
        return F.sum(F.broadcast_mul(input, scores), axis=0)


class FCLayerWithAttention(nn.HybridBlock):

    def __init__(
        self, num_layers, hidden_size=512, activation='relu', output_size=1,
        prefix='attfc_', **kwargs
    ):
        super().__init__(prefix=prefix, **kwargs)
        with self.name_scope():
            self.attention = _HybridConcurrent(axis=1, prefix='att_')
            with self.attention.name_scope():
                for i in range(output_size):
                    self.attention.add(AttentionCell(prefix=f'o{i}_'))
            self.fc_layer = FCLayer(
                num_layers, hidden_size=hidden_size,
                activation=activation, output_size=1
            )

    def hybrid_forward(self, F, input, mask):
        """
        input: shape(seq_length, batch_size, dim)
        mask: shape(seq_length, batch_size)
        """
        # (batch_size, output_size, dim)
        output = self.attention(input, mask)
        return F.squeeze(self.fc_layer(output), axis=-1)


class TextCNN(nn.HybridBlock):

    def __init__(
        self, embed_size=100,
        num_filters=(25, 50, 75, 100), ngram_filter_sizes=(1, 2, 3, 4),
        conv_layer_activation='relu', num_highways=1, dropout=0, output_size=1,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='relu', **kwargs
    ):
        super().__init__(**kwargs)
        self._dropout = dropout
        with self.name_scope():
            if self._dropout:
                self.input_dropout = nn.Dropout(dropout)
                self.cnn_dropout = nn.Dropout(dropout)
            self.cnn_layer = ConvolutionalEncoder(
                embed_size, num_highways=num_highways,
                num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                prefix='cnn_'
            )
            self.fc_layer = FCLayer(
                num_layers=num_fc_layers, hidden_size=fc_hidden_size,
                activation=fc_activation, output_size=output_size
            )

    def hybrid_forward(self, F, input, mask):
        """
        input: shape(seq_length, batch_size, dim)
        mask: shape(seq_length, batch_size)
        """
        if self._dropout:
            cnn_output = self.cnn_dropout(
                self.cnn_layer(self.input_dropout(input), mask)
            )
        else:
            cnn_output = self.cnn_layer(input, mask)
        return self.fc_layer(cnn_output)


class TextRNN(nn.HybridBlock):

    def __init__(
        self, num_rnn_layers=1, projection_size=128, hidden_size=1024,
        cell_clip=3, projection_clip=3, dropout=0.0, dense_connection=None,
        output_size=1, num_fc_layers=2, fc_hidden_size=512,
        fc_activation='relu', **kwargs
    ):
        super().__init__(**kwargs)
        self._dense_connection = dense_connection
        with self.name_scope():
            self.rnn_layer = BiLSTM(
                num_rnn_layers, projection_size, hidden_size=hidden_size,
                cell_clip=cell_clip, dropout=dropout, prefix='rnn_'
            )
            if self._dense_connection == 'attention':
                fc_layer = FCLayerWithAttention
            else:
                fc_layer = FCLayer
            self.fc_layer = fc_layer(
                num_layers=num_fc_layers, hidden_size=fc_hidden_size,
                activation=fc_activation, output_size=output_size
            )

    def hybrid_forward(self, F, input, mask):
        """
        input: shape(seq_length, batch_size, dim)
        mask: shape(seq_length, batch_size)
        """
        rnn_output, _ = self.rnn_layer(input, states=None, mask=mask)
        if self._dense_connection == 'attention':
            return self.fc_layer(rnn_output, mask)
        elif self._dense_connection == 'last':
            forward_output, backward_output = F.split(
                rnn_output, axis=-1, num_outputs=2
            )
            rnn_output = F.concat(
                F.SequenceLast(
                    forward_output, sequence_length=F.sum(mask, axis=0),
                    use_sequence_length=True
                ),
                F.squeeze(
                    F.slice_axis(backward_output, axis=0, begin=0, end=1),
                    axis=0
                ),
                dim=-1
            )
        return self.fc_layer(rnn_output)


class TextRCNN(nn.HybridBlock):

    def __init__(
        self, num_rnn_layers=1, projection_size=128, hidden_size=1024,
        cell_clip=3, projection_clip=3, dropout=0.0, kmax=2, output_size=1,
        num_fc_layers=1, fc_hidden_size=512, fc_activation='relu', **kwargs
    ):
        super().__init__(**kwargs)
        self._kmax = kmax
        self._dropout = dropout
        with self.name_scope():
            if self._dropout:
                self.input_dropout = nn.Dropout(dropout)
            self.rnn_layer = BiLSTM(
                num_rnn_layers, projection_size, hidden_size=hidden_size,
                cell_clip=cell_clip, dropout=dropout, prefix='rnn_'
            )
            self.dense = nn.Dense(
                fc_hidden_size, flatten=False,
                activation='tanh', prefix='dense_'
            )
            self.fc_layer = FCLayer(
                num_layers=num_fc_layers, hidden_size=fc_hidden_size,
                activation=fc_activation, output_size=output_size
            )

    def hybrid_forward(self, F, input, mask):
        """
        input: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        rnn_output, _ = self.rnn_layer(input, states=None, mask=mask)
        if self._dropout:
            input = self.input_dropout(input)
        mixed_output = self.dense(F.SequenceMask(
            F.concat(input, rnn_output, dim=-1),
            sequence_length=F.sum(mask, axis=0),
            use_sequence_length=True
        ))
        kmaxpooling_output = F.topk(
            mixed_output, axis=0, ret_typ='value', k=self._kmax
        )
        return self.fc_layer(
            F.flatten(F.transpose(kmaxpooling_output, axes=(1, 0, 2)))
        )
