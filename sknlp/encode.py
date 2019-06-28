from mxnet.gluon import nn

from gluonnlp.model import ConvolutionalEncoder

from .module import BiLSTMWithClip, BaseTransformerEncoder


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
                    activation = activation if i == num_layers - 2 else 'relu'
                    self.net.add(nn.Dense(
                        hidden_size, flatten=False, prefix=f'layer{i}_'
                    ))
                    self.net.add(nn.BatchNorm(axis=-1))
                    self.net.add(nn.Activation(activation))

    def hybrid_forward(self, F, inputs):
        return self.net(inputs)


class TextCNN(nn.HybridBlock):

    def __init__(
        self, embed_size=100,
        num_filters=(25, 50, 75, 100), ngram_filter_sizes=(1, 2, 3, 4),
        conv_layer_activation='tanh', num_highway=1, dropout=0, output_size=1,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='relu', **kwargs
    ):
        super().__init__(**kwargs)
        with self.name_scope():
            self.input_dropout = nn.Dropout(dropout)
            self.cnn_layer = ConvolutionalEncoder(
                embed_size=embed_size, num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                output_size=None, num_highway=num_highway, prefix='cnn_'
            )
            self.cnn_dropout = nn.Dropout(dropout)
            self.fc_layer = FCLayer(
                num_layers=num_fc_layers, hidden_size=fc_hidden_size,
                activation=fc_activation, output_size=output_size
            )

    def hybrid_forward(self, F, inputs, mask):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        cnn_output = self.cnn_dropout(
            self.cnn_layer(self.input_dropout(inputs), mask)
        )
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
            self.rnn_layer = BiLSTMWithClip(
                num_rnn_layers, projection_size, hidden_size=hidden_size,
                cell_clip=cell_clip, projection_clip=projection_clip,
                dropout=dropout, prefix='rnn_'
            )
            self.fc_layer = FCLayer(
                num_layers=num_fc_layers, hidden_size=fc_hidden_size,
                activation=fc_activation, output_size=output_size
            )

    def hybrid_forward(self, F, inputs, mask):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        rnn_output, _ = self.rnn_layer(inputs, states=None, mask=mask)
        if self._dense_connection == 'last':
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
        with self.name_scope():
            self.rnn_layer = BiLSTMWithClip(
                num_rnn_layers, projection_size, hidden_size=hidden_size,
                cell_clip=cell_clip, projection_clip=projection_clip,
                dropout=dropout, prefix='rnn_'
            )
            self.dense = nn.Dense(
                fc_hidden_size, flatten=False,
                activation='tanh', prefix='dense_'
            )
            self.fc_layer = FCLayer(
                num_layers=num_fc_layers, hidden_size=fc_hidden_size,
                activation=fc_activation, output_size=output_size
            )

    def hybrid_forward(self, F, inputs, mask):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        rnn_output, _ = self.rnn_layer(inputs, states=None, mask=mask)
        mixed_output = self.dense(F.concat(inputs, rnn_output, dim=-1))
        kmaxpooling_output = F.topk(
            mixed_output, axis=0, ret_typ='value', k=self._kmax
        )
        return self.fc_layer(
            F.flatten(F.transpose(kmaxpooling_output, axes=(1, 0, 2)))
        )


class TextTransformer(nn.HybridBlock):

    def __init__(
        self, attention_cell='multi_head', num_layers=2,
        units=512, hidden_size=2048, max_length=50,
        num_heads=4, scaled=True, dropout=0.0,
        use_residual=True, output_attention=False,
        weight_initializer=None, bias_initializer='zeros',
        prefix=None, params=None, output_size=1,
        num_fc_layers=1, fc_hidden_size=512, fc_activation='relu', **kwargs
    ):
        super().__init__(prefix=prefix)
        with self.name_scope():
            self.transformer_layer = BaseTransformerEncoder(
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
                prefix=prefix, params=params,
            )
            self.fc_layer = FCLayer(
                num_layers=num_fc_layers, hidden_size=fc_hidden_size,
                activation=fc_activation, output_size=output_size
            )

    def hybrid_forward(self, F, inputs, mask, steps=None):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        transformer_output = self.transformer_layer(inputs, steps, mask)
        valid_length = F.sum(mask, axis=1)
        sentence_vec = F.broadcast_div(
            F.sum(
                F.SequenceMask(
                    transformer_output, sequence_length=valid_length,
                    use_sequence_length=True, axis=1
                ),
                axis=1
            ),
            F.sqrt(valid_length)
        )
        return self.fc_layer(sentence_vec)
