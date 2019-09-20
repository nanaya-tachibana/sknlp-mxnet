import mxnet as mx
from mxnet.gluon import nn, rnn
from gluonnlp.model import LSTMPCellWithClip, Highway
from mxnet.gluon.contrib.rnn import VariationalDropoutCell


class BiLSTM(nn.HybridBlock):

    def __init__(
        self, num_layers, projection_size, layout='TNC',
        hidden_size=1024, cell_clip=3, dropout=0.5,
        i2h_weight_initializer=mx.initializer.Orthogonal(),
        h2h_weight_initializer=mx.initializer.Orthogonal(),
        h2r_weight_initializer=mx.initializer.Orthogonal(),
        i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
        input_size=0, return_all_layers=False, prefix='bilstm_',
        **kwargs
    ):
        super().__init__(prefix=prefix, **kwargs)
        self._num_layers = num_layers
        self._dropout = dropout
        self._layout = layout
        self._return_all_layers = return_all_layers
        self._cell_clip = cell_clip
        with self.name_scope():
            self.input_dropout = nn.Dropout(dropout)
            self.forward_layers = []
            self.backward_layers = []
            for layers in (self.forward_layers, self.backward_layers):
                for i in range(num_layers):
                    layer = rnn.LSTM(
                        hidden_size, input_size=input_size,
                        i2h_weight_initializer=i2h_weight_initializer,
                        h2h_weight_initializer=h2h_weight_initializer,
                        h2r_weight_initializer=h2r_weight_initializer,
                        state_clip_nan=True,
                        # mkldnn not support projection and state clip for now
                        # projection_size=projection_size,
                        # state_clip_min=-cell_clip, state_clip_max=cell_clip
                    )
                    layers.append(layer)
                    self.register_child(layer)
            if self._dropout:
                self.forward_dropout_layers = []
                self.backward_dropout_layers = []
                for layers in (
                    self.forward_dropout_layers, self.backward_dropout_layers
                ):
                    for i in range(num_layers):
                        layer = nn.Dropout(dropout)
                        layers.append(layer)
                        self.register_child(layer)

    def begin_state(self, batch_size=0, func=mx.ndarray.zeros, **kwargs):
        return [
            [
                layer.begin_state(
                    batch_size=batch_size, func=func, **kwargs
                ) for layer in layers
            ] for layers in (self.forward_layers, self.backward_layers)
        ]

    def __call__(self, input, states=None, mask=None, **kwargs):
        if states is None:
            if isinstance(input, mx.ndarray.NDArray):
                batch_size = input.shape[self._layout.find('N')]
                states = self.begin_state(
                    batch_size=batch_size, func=mx.ndarray.zeros,
                    ctx=input.context, dtype=input.dtype
                )
            else:
                states = self.begin_state(func=mx.symbol.zeros)
        return super().__call__(input, states, mask, **kwargs)

    def hybrid_forward(self, F, input, states=None, mask=None):
        forward_states, backward_states = states
        sequence_length = F.sum(mask, axis=0)
        forward_outputs = []
        backward_outputs = []
        for layer_index in range(self._num_layers):
            if layer_index == 0:
                forward_layer_input = input
                backward_layer_input = F.SequenceReverse(
                    input, sequence_length=sequence_length,
                    use_sequence_length=True, axis=0
                )
            else:
                forward_layer_input = forward_outputs[layer_index - 1]
                backward_layer_input = backward_outputs[layer_index - 1]

            output, states = self.forward_layers[layer_index](
                forward_layer_input, forward_states[layer_index]
            )
            forward_states[layer_index] = states
            if self._dropout:
                output = self.forward_dropout_layers[layer_index](output)
            forward_outputs.append(output)

            output, states = self.backward_layers[layer_index](
                backward_layer_input, backward_states[layer_index]
            )
            backward_states[layer_index] = states
            if self._dropout:
                output = self.backward_dropout_layers[layer_index](output)
            backward_outputs.append(output)

        backward_outputs = [
            F.SequenceReverse(
                output, sequence_length=sequence_length,
                use_sequence_length=True, axis=0
            ) for output in backward_outputs
        ]
        if self._return_all_layers:
            out = F.concat(
                F.stack(*forward_outputs, axis=0),
                F.stack(*backward_outputs, axis=0),
                dim=-1
            )
        else:
            out = F.concat(forward_outputs[-1], backward_outputs[-1], dim=-1)
        return out, (forward_states, backward_states)


class BiLSTMWithClip(nn.HybridBlock):

    def __init__(
        self, num_layers, projection_size, layout='TNC',
        hidden_size=1024, cell_clip=3, projection_clip=3, dropout=0.5,
        i2h_weight_initializer=mx.initializer.Orthogonal(),
        h2h_weight_initializer=mx.initializer.Orthogonal(),
        h2r_weight_initializer=mx.initializer.Orthogonal(),
        i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
        input_size=0, return_all_layers=False, prefix='bilstm_', **kwargs
    ):
        super().__init__(prefix=prefix, **kwargs)
        self._num_layers = num_layers
        self._return_all_layers = return_all_layers
        self._projection_size = projection_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._layout = layout
        with self.name_scope():
            self.forward_layers = rnn.HybridSequentialRNNCell(
                prefix='forward_'
            )
            self.backward_layers = rnn.HybridSequentialRNNCell(
                prefix='backward_'
            )
            for layers in (self.forward_layers, self.backward_layers):
                lstm_input_size = input_size
                with layers.name_scope():
                    for i in range(num_layers):
                        stack_cell = rnn.HybridSequentialRNNCell(
                            prefix=f'{i}_'
                        )
                        with stack_cell.name_scope():
                            cell = LSTMPCellWithClip(
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
                            )
                            if dropout:
                                if i != num_layers - 1:
                                    cell = VariationalDropoutCell(
                                        cell,
                                        drop_inputs=dropout,
                                        drop_states=dropout
                                    )
                                else:
                                    cell = VariationalDropoutCell(
                                        cell,
                                        drop_inputs=dropout,
                                        drop_states=dropout,
                                        drop_outputs=dropout
                                    )
                            stack_cell.add(cell)
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
                states.append([
                    func(
                        name='%s%sph0_%d' % (self.prefix, name, i),
                        **projection_info
                    ),
                    func(
                        name='%s%shh0_%d' % (self.prefix, name, i),
                        **hidden_info
                    )
                ])
        return forward_states, backward_states

    def state_info(self, batch_size=0):
        if self._projection_size is None:
            return [
                [
                    {'shape': (batch_size, self._hidden_size)},
                    {'shape': (batch_size, self._hidden_size)}
                ] for _ in range(self._num_layers)
            ]
        else:
            return [
                [
                    {'shape': (batch_size, self._projection_size)},
                    {'shape': (batch_size, self._hidden_size)}
                ] for _ in range(self._num_layers)
            ]

    def __call__(self, input, states=None, mask=None, **kwargs):
        if states is None:
            if isinstance(input, mx.ndarray.NDArray):
                batch_size = input.shape[self._layout.find('N')]
                states = self.begin_state(
                    batch_size=batch_size, func=mx.ndarray.zeros,
                    ctx=input.context, dtype=input.dtype
                )
            else:
                states = self.begin_state(func=mx.symbol.zeros)
        return super().__call__(input, states, mask, **kwargs)

    def hybrid_forward(self, F, input, states=None, mask=None):
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Defines the forward computation for cache cell. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.
        Parameters
        ----------
        input : NDArray
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
        forward_states, backward_states = states
        if mask is not None:
            sequence_length = F.sum(mask, axis=0)

        forward_outputs = []
        backward_outputs = []

        for layer_index in range(self._num_layers):
            if layer_index == 0:
                layer_input = input
            else:
                layer_input = F.concat(
                    forward_outputs[layer_index - 1],
                    backward_outputs[layer_index - 1],
                    dim=-1
                )
            if self._dropout:
                self.forward_layers[layer_index][0].reset()
            output, forward_states[layer_index] = F.contrib.foreach(
                self.forward_layers[layer_index], layer_input,
                forward_states[layer_index]
            )
            forward_outputs.append(output)

            if mask is not None:
                layer_input = F.SequenceReverse(
                    layer_input, sequence_length=sequence_length,
                    use_sequence_length=True, axis=0
                )
            else:
                layer_input = F.SequenceReverse(layer_input, axis=0)
            if self._dropout:
                self.backward_layers[layer_index][0].reset()
            output, backward_states[layer_index] = F.contrib.foreach(
                self.backward_layers[layer_index], layer_input,
                backward_states[layer_index]
            )
            if mask is not None:
                backward_output = F.SequenceReverse(
                    output, sequence_length=sequence_length,
                    use_sequence_length=True, axis=0
                )
            else:
                backward_output = F.SequenceReverse(output, axis=0)
            backward_outputs.append(backward_output)
        if self._return_all_layers:
            out = F.concat(
                F.stack(*forward_outputs, axis=0),
                F.stack(*backward_outputs, axis=0),
                dim=-1
            )
        else:
            out = F.concat(forward_outputs[-1], backward_outputs[-1], dim=-1)
        return out, [forward_states, backward_states]


class ConvEncoder(nn.HybridBlock):

    def __init__(
        self, input_size, num_highways=1, projection_size=0,
        num_filters=(25, 50, 75, 100, 125, 150),
        ngram_filter_sizes=(1, 2, 3, 4, 5, 6),
        conv_layer_activation='relu', **kwargs
    ):
        super().__init__(**kwargs)
        with self.name_scope():
            self.convs = []
            maxpool_output_size = 0
            with self.convs.name_scope():
                for num_filter, ngram_size in zip(num_filters,
                                                  ngram_filter_sizes):
                    seq = nn.HybridSequential()
                    seq.add(nn.Conv1D(
                        in_channels=input_size,
                        channels=num_filter,
                        kernel_size=ngram_size,
                        use_bias=True
                    ))
                    if conv_layer_activation is not None:
                        seq.add(nn.Activation(conv_layer_activation))
                    seq.add(nn.BatchNorm(axis=1))
                    seq.add(mx.gluon.nn.HybridLambda(
                        lambda F, x: F.max(x, axis=2)
                    ))
                    self.convs.append(seq)
                    self.register_child(seq)
                    maxpool_output_size += num_filter

            self.highways = Highway(
                maxpool_output_size, num_highways,
            ) if num_highways > 0 else None
            self.project = nn.Dense(
                units=projection_size,
                in_units=maxpool_output_size,
            ) if projection_size > 0 else None

    def hybrid_forward(self, F, input, mask=None):
        if mask is not None:
            input = F.broadcast_mul(input, mask.expand_dims(-1))
        input = F.transpose(input, axes=(1, 2, 0))
        output = F.concat(*[conv(input) for conv in self.convs], dim=-1)
        if self.highways is not None:
            output = self.highways(output)
        if self.project is not None:
            output = self.project(output)
        return output
