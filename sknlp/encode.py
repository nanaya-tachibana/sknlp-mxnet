import os
import tempfile
import shutil
import mxnet as mx
from mxnet.gluon import nn, rnn

from gluonnlp.model import LSTMPCellWithClip
from gluonnlp.model import ConvolutionalEncoder


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


class BiLSTMWithClip(nn.HybridBlock):

    def __init__(
        self, num_layers, projection_size, hidden_size=1024,
        cell_clip=3, projection_clip=3, skip_connection=True, dropout=0.5,
        i2h_weight_initializer=None, h2h_weight_initializer=None,
        h2r_weight_initializer=None, i2h_bias_initializer='zeros',
        h2h_bias_initializer='zeros', input_size=0, return_all_layers=False,
        prefix='bilstm_', **kwargs
    ):
        super().__init__(prefix=prefix, **kwargs)
        self._num_layers = num_layers
        self._return_all_layers = return_all_layers
        self._projection_size = projection_size
        self._hidden_size = hidden_size
        self._layout = 'LNC'
        with self.name_scope():
            self.forward_layers = rnn.HybridSequentialRNNCell(prefix='forward_')
            self.backward_layers = rnn.HybridSequentialRNNCell(prefix='backward_')
            for layers in (self.forward_layers, self.backward_layers):
                lstm_input_size = input_size
                with layers.name_scope():
                    for i in range(num_layers):
                        stack_cell = rnn.HybridSequentialRNNCell(
                            prefix=f'{i}_'
                        )
                        with stack_cell.name_scope():
                            stack_cell.add(LSTMPCellWithClip(
                                hidden_size,
                                projection_size,
                                cell_clip=cell_clip,
                                projection_clip=projection_clip,
                                i2h_weight_initializer=i2h_weight_initializer,
                                h2h_weight_initializer=h2h_weight_initializer,
                                h2r_weight_initializer=h2r_weight_initializer,
                                i2h_bias_initializer=i2h_bias_initializer,
                                h2h_bias_initializer=h2h_bias_initializer,
                                input_size=lstm_input_size,
                            ))
                            if i != num_layers - 1 and dropout != 0:
                                stack_cell.add(rnn.DropoutCell(dropout))
                        layers.add(stack_cell)
                        lstm_input_size = projection_size * 2

    def begin_state(self, batch_size=0, func=mx.ndarray.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        batch_size: int
            Only required for `NDArray` API. Size of the batch ('N' in layout).
            Dimension of the input.
        func : callable, default `ndarray.zeros`
            Function for creating initial state.

            For Symbol API, func can be `symbol.zeros`, `symbol.uniform`,
            `symbol.var` etc. Use `symbol.var` if you want to directly
            feed input as states.

            For NDArray API, func can be `ndarray.zeros`, `ndarray.ones`, etc.

        **kwargs :
            Additional keyword arguments passed to func. For example
            `mean`, `std`, `dtype`, etc.

        Returns
        -------
        states : nested list of Symbol
            Starting states for the first RNN step.
        """
        forward_states = []
        backward_states = []
        for name, states in (('f', forward_states), ('b', backward_states)):
            for i, info in enumerate(self.state_info(batch_size)):
                projection_info, hidden_info = info
                projection_info.update(kwargs)
                hidden_info.update(kwargs)
                states.append((
                    func(
                        name='%s%sph0_%d' % (self.prefix, name, i),
                        **projection_info
                    ),
                    func(
                        name='%s%shh0_%d' % (self.prefix, name, i),
                        **hidden_info
                    )
                ))
        return (forward_states, backward_states)

    def state_info(self, batch_size=0):
        if self._projection_size is None:
            return [
                (
                    {
                        'shape': (batch_size, self._hidden_size),
                    },
                    {
                        'shape': (batch_size, self._hidden_size),
                    }
                ) for _ in range(self._num_layers)
            ]
        else:
            return [
                (
                    {
                        'shape': (batch_size, self._projection_size),
                    },
                    {
                        'shape': (batch_size, self._hidden_size),
                    }
                ) for _ in range(self._num_layers)
            ]

    def __call__(self, inputs, states=None, mask=None, **kwargs):
        if states is None:
            if isinstance(inputs, mx.ndarray.NDArray):
                batch_size = inputs.shape[self._layout.find('N')]
                states = self.begin_state(
                    batch_size=batch_size, func=mx.ndarray.zeros,
                    ctx=inputs.context, dtype=inputs.dtype
                )
            else:
                states = self.begin_state(func=mx.symbol.zeros)
        return super().__call__(inputs, states, mask, **kwargs)

    def hybrid_forward(self, F, inputs, states=None, mask=None):
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Defines the forward computation for cache cell. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.
        Parameters
        ----------
        inputs : NDArray
            The input data layout='TNC'.
        states : Tuple[List[List[NDArray]]]
            The states. including:
            states[0] indicates the states used in forward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
            states[1] indicates the states used in backward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
        Returns
        --------
        out: NDArray
            The output data with shape (num_layers, seq_len, batch_size, 2*input_size).
        [states_forward, states_backward] : List
            Including:
            states_forward: The out states from forward layer,
            which has the same structure with *states[0]*.
            states_backward: The out states from backward layer,
            which has the same structure with *states[1]*.
        """
        states_forward, states_backward = states
        if mask is not None:
            sequence_length = F.sum(mask, axis=0)

        outputs_forward = []
        outputs_backward = []

        for layer_index in range(self._num_layers):
            if layer_index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = F.concat(
                    outputs_forward[layer_index - 1],
                    outputs_backward[layer_index - 1],
                    dim=-1,
                )
            output, states_forward[layer_index] = F.contrib.foreach(
                self.forward_layers[layer_index],
                layer_inputs,
                states_forward[layer_index]
            )
            outputs_forward.append(output)

            if mask is not None:
                layer_inputs = F.SequenceReverse(
                    layer_inputs,
                    sequence_length=sequence_length,
                    use_sequence_length=True,
                    axis=0
                )
            else:
                layer_inputs = F.SequenceReverse(layer_inputs, axis=0)
            output, states_backward[layer_index] = F.contrib.foreach(
                self.backward_layers[layer_index],
                layer_inputs,
                states_backward[layer_index]
            )
            if mask is not None:
                backward_output = F.SequenceReverse(
                    output,
                    sequence_length=sequence_length,
                    use_sequence_length=True,
                    axis=0
                )
            else:
                backward_output = F.SequenceReverse(output, axis=0)
            outputs_backward.append(backward_output)
        if self._return_all_layers:
            out = F.concat(
                F.stack(*outputs_forward, axis=0),
                F.stack(*outputs_backward, axis=0),
                dim=-1
            )
        else:
            out = F.concat(outputs_forward[-1], outputs_backward[-1], dim=-1)
        return out, [states_forward, states_backward]


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
        self, embed_size=100,
        num_filters=(25, 50, 75, 100), ngram_filter_sizes=(1, 2, 3, 4),
        conv_layer_activation='tanh', num_highway=1, output_size=1,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='relu', **kwargs
    ):
        super().__init__(**kwargs)
        with self.name_scope():
            self.cnn_layer = ConvolutionalEncoder(
                embed_size=embed_size, num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                output_size=None, num_highway=num_highway, prefix='cnn_'
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
        return self.fc_layer(self.cnn_layer(inputs, mask))


class TextRNN(SequenceEncode):

    def __init__(
        self, num_rnn_layers=1, projection_size=128, hidden_size=1024,
        cell_clip=3, projection_clip=3, dropout=0.0, dense_connection='last',
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
                    F.slice_axis(
                        backward_output, axis=0, begin=0, end=1
                    ), axis=0
                ),
                dim=-1
            )
        return self.fc_layer(rnn_output)


class TextRCNN(SequenceEncode):

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

    def hybrid_forward(self, F, inputs, mask=None):
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
