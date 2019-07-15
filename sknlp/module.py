import mxnet as mx
from mxnet.gluon import nn, rnn
from gluonnlp.model import LSTMPCellWithClip
from gluonnlp.model.transformer import (
    _get_layer_norm, TransformerEncoderCell, _position_encoding_init
)
from gluonnlp.model.bert import BERTEncoderCell
from mxnet.gluon.contrib.rnn import VariationalDropoutCell


class BiLSTM(nn.HybridBlock):

    def __init__(
        self, num_layers, projection_size, layout='TNC',
        hidden_size=1024, cell_clip=3, dropout=0.5,
        i2h_weight_initializer=mx.initializer.Orthogonal(),
        h2h_weight_initializer=mx.initializer.Orthogonal(),
        i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
        input_size=0, return_all_layers=False, prefix='bilstm_', **kwargs
    ):
        super().__init__(prefix=prefix, **kwargs)
        self._num_layers = num_layers
        self._dropout = dropout
        self._layout = layout
        self._return_all_layers = return_all_layers
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
                        projection_size=projection_size, state_clip_nan=True,
                        state_clip_min=-cell_clip, state_clip_max=cell_clip
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


class BaseTransformerEncoder(nn.HybridBlock):
    """Base Structure of the Transformer Encoder.
    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    use_residual : bool
    output_attention: bool, default False
        Whether to output the attention weights
    output_all_encodings: bool, default False
        Whether to output encodings of all encoder's cells, or only the last one
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the input.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    positional_weight: str, default 'sinusoidal'
        Type of positional embedding. Can be 'sinusoidal', 'learned'.
        If set to 'sinusoidal', the embedding is initialized as sinusoidal values and keep constant.
    use_bert_encoder : bool, default False
        Whether to use BERTEncoderCell and BERTLayerNorm. Set to True for pre-trained BERT model
    use_layer_norm_before_dropout: bool, default False
        Before passing embeddings to attention cells, whether to perform `layernorm -> dropout` or
        `dropout -> layernorm`. Set to True for pre-trained BERT models.
    scale_embed : bool, default True
        Scale the input embeddings by sqrt(embed_size). Set to False for pre-trained BERT models.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """

    def __init__(
        self, input_size=300, attention_cell='multi_head', num_layers=2,
        units=512, hidden_size=2048, max_length=50,
        num_heads=4, scaled=True, dropout=0.0,
        use_residual=True, output_attention=False, output_all_encodings=False,
        weight_initializer=None, bias_initializer='zeros',
        positional_weight='sinusoidal', use_bert_encoder=False,
        use_layer_norm_before_dropout=False, scale_embed=True,
        prefix=None, params=None
    ):
        super(BaseTransformerEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._input_size = input_size
        self._num_layers = num_layers
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        self._use_layer_norm_before_dropout = use_layer_norm_before_dropout
        self._scale_embed = scale_embed
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = _get_layer_norm(use_bert_encoder, units)
            self.position_weight = self._get_positional(
                positional_weight, max_length, units, weight_initializer
            )
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                cell = self._get_encoder_cell(
                    use_bert_encoder, units, hidden_size, num_heads,
                    attention_cell, weight_initializer, bias_initializer,
                    dropout, use_residual, scaled, output_attention, i
                )
                self.transformer_cells.add(cell)

    def _get_positional(self, weight_type, max_length, units, initializer):
        if weight_type == 'sinusoidal':
            encoding = _position_encoding_init(max_length, units)
            position_weight = self.params.get_constant('const', encoding)
        elif weight_type == 'learned':
            position_weight = self.params.get(
                'position_weight', shape=(max_length, units),
                init=initializer
            )
        else:
            raise ValueError(
                'Unexpected value for argument position_weight: %s'
                % (position_weight)
            )
        return position_weight

    def _get_encoder_cell(
        self, use_bert, units, hidden_size, num_heads, attention_cell,
        weight_initializer, bias_initializer, dropout, use_residual,
        scaled, output_attention, i
    ):
        cell = BERTEncoderCell if use_bert else TransformerEncoderCell
        return cell(units=units, hidden_size=hidden_size,
                    num_heads=num_heads, attention_cell=attention_cell,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer,
                    dropout=dropout, use_residual=use_residual,
                    scaled=scaled, output_attention=output_attention,
                    prefix='transformer%d_' % i
                    )

    def hybrid_forward(self, F, input, steps, mask, position_weight=None):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input : NDArray or Symbol, Shape(batch_size, length, C_in)
        states : list of NDArray or Symbol
        valid_length : NDArray or Symbol
        position_weight : NDArray or Symbol
        Returns
        -------
        output : NDArray or Symbol, or List[NDArray] or List[Symbol]
            If output_all_encodings flag is False, then the output of the last encoder.
            If output_all_encodings flag is True, then the list of all output of all encoders.
            In both cases, shape of the tensor(s) is/are (batch_size, length, C_out)
        additional_output : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)
        """
        # Positional Encoding
        if self._scale_embed:
            input = input * F.sqrt(self._input_size)

        positional_embed = F.Embedding(
            steps, position_weight, self._max_length, self._units
        )
        input = F.broadcast_add(
            input, F.expand_dims(positional_embed, axis=0)
        )
        if self._dropout:
            if self._use_layer_norm_before_dropout:
                input = self.layer_norm(input)
                input = self.dropout_layer(input)
            else:
                input = self.dropout_layer(input)
                input = self.layer_norm(input)
        else:
            input = self.layer_norm(input)
        output = input

        valid_length = None
        if mask is not None:
            valid_length = F.sum(mask, axis=1)
        all_encodings_output = []
        additional_output = []
        for cell in self.transformer_cells:
            output, attention_weights = cell(input, mask)
            input = output
            if self._output_all_encodings:
                if valid_length is not None:
                    valid_length = F.sum(mask, axis=1)
                    output = F.SequenceMask(
                        output, sequence_length=valid_length,
                        use_sequence_length=True, axis=1
                    )
                all_encodings_output.append(output)

            if self._output_attention:
                additional_output.append(attention_weights)

        if valid_length is not None:
            output = F.SequenceMask(
                output, sequence_length=valid_length,
                use_sequence_length=True, axis=1
            )

        if self._output_all_encodings:
            return all_encodings_output, additional_output
        else:
            return output, additional_output
