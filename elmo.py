# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Bidirectional LM encoder."""
__all__ = ['BiLMEncoder']

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import rnn

from base import DeepModel


class BiLMEncoder(gluon.HybridBlock):
    """Bidirectional LM encoder.

    We implement the encoder of the biLM proposed in the following work::

        @inproceedings{Peters:2018,
        author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark,
        Christopher and Lee, Kenton and Zettlemoyer, Luke},
        title={Deep contextualized word representations},
        booktitle={Proc. of NAACL},
        year={2018}
        }

    Parameters
    ----------
    cell_type : str
        The type of RNN cell to use. Options are 'rnn_tanh', 'rnn_relu', 'lstm', 'gru'.
    num_layers : int
        The number of RNN cells in the encoder.
    input_size : int
        The initial input size of in the RNN cell.
    hidden_size : int
        The hidden size of the RNN cell.
    dropout : float
        The dropout rate to use for encoder output.
    residual_connection : bool
        Whether to add skip connections (add RNN cell input to output)
    """

    def __init__(self, cell_type='lstm', num_layers=1,
                 input_size=200, hidden_size=512,
                 dropout=0.0, residual_connection=True, **kwargs):
        super().__init__(**kwargs)

        self._cell_type = cell_type
        self._num_layers = num_layers
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._residual_connection = residual_connection

        with self.name_scope():
            self.weight = self.params.get('weight',
                                          shape=(input_size, rnn_input_size),
                                          init=mx.init.Uniform(0.1))
            self.bias = self.params.get('bias',
                                        shape=(vocab_size,),
                                        init=mx.init.Zero())

            rnn_input_size = self._input_size
            self.forward_layers = rnn.HybridSequentialRNNCell()
            with self.forward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    l_cell = self._get_cell(rnn_input_size,
                                            residual_connection=False
                                            if layer_index == 0
                                            else residual_connection,
                                            dropout=0
                                            if layer_index == num_layers - 1
                                            else dropout)
                    self.forward_layers.add(l_cell)
                    rnn_input_size = hidden_size

            rnn_input_size = self._input_size
            self.backward_layers = rnn.HybridSequentialRNNCell()
            with self.backward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    r_cell = self._get_cell(rnn_input_size,
                                            residual_connection=False
                                            if layer_index == 0
                                            else residual_connection,
                                            dropout=0
                                            if layer_index == num_layers - 1
                                            else dropout)
                    self.backward_layers.add(r_cell)
                    rnn_input_size = hidden_size

    def _get_cell(self, input_size, residual_connection=False, dropout=0.0):
        cell = rnn.HybirdSequentialRNNCell()
        with cell.name_scope():
            if self._cell_type == 'rnn_relu':
                cell.add(rnn.RNNCell(self._hidden_size, 'relu',
                                     input_size=input_size))
            elif self._cell_type == 'rnn_tanh':
                cell.add(rnn.RNNCell(self._hidden_size, 'tanh',
                                     input_size=input_size))
            elif self._cell_type == 'lstm':
                cell.add(rnn.LSTMCell(self._hidden_size,
                                      input_size=input_size))
            elif self._cell_type == 'gru':
                cell.add(rnn.GRUCell(self._hidden_size, input_size=input_size))

            if residual_connection:
                cell.add(rnn.ResidualCell(cell))

            if dropout:
                cell.add(rnn.DropoutCell(cell))
        return cell

    def begin_state(self, func, **kwargs):
        return ([self.forward_layers[0][0].begin_state(func=func, **kwargs)
                 for _ in range(self._num_layers)],
                [self.backward_layers[0][0].begin_state(func=func, **kwargs)
                 for _ in range(self._num_layers)])

    def hybrid_forward(self, F, inputs, states, mask=None):
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
            sequence_length = mask.sum(axis=1)

        outputs_forward = []
        outputs_backward = []

        for layer_index in range(self._num_layers):
            if layer_index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = outputs_forward[layer_index - 1]
            output, states_forward[layer_index] = F.contrib.foreach(
                self.forward_layers[layer_index], layer_inputs,
                states_forward[layer_index])
            outputs_forward.append(output)

            if layer_index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = outputs_backward[layer_index - 1]

            if mask is not None:
                layer_inputs = F.SequenceReverse(
                    layer_inputs, sequence_length=sequence_length,
                    use_sequence_length=True, axis=0)
            else:
                layer_inputs = F.SequenceReverse(layer_inputs, axis=0)
            output, states_backward[layer_index] = F.contrib.foreach(
                self.backward_layers[layer_index], layer_inputs,
                states_backward[layer_index])
            if mask is not None:
                output = F.SequenceReverse(
                    output, sequence_length=sequence_length,
                    use_sequence_length=True, axis=0)
            else:
                output = F.SequenceReverse(output, axis=0)
            outputs_backward.append(output)
        out = F.concat(*[F.stack(*outputs_forward, axis=0),
                         F.stack(*outputs_backward, axis=0)], dim=-1)

        return out, [states_forward, states_backward]


class BiLMLoss(gluon.HybridBlock):

    def __init__(self, vocab_size, embed_size, **kwargs):
        super().__init__(**kwargs)

        self._vocab_size = vocab_size
        self._embed_size = embed_size

    def hybrid_forward(self):
        pass


class Elmo(DeepModel):

    def __init__(self, cell_type='lstm', num_layers=1,
                 input_size=200, hidden_size=512,
                 dropout=0.0, residual_connection=True,
                 model=None, embed_weight=None):
        self.meta = {
            'cell_type': 'lstm',
            'num_layers': 1,
            'input_size': 200,
            'hidden_size': 512,
            'dropout': 0.0,
            'residual_connection': True,
        }
        initialized = False if model is None else True
        if model is None:
            model = None
        super().__init__(vocab, model)
        self.encoder = self.model.encoder
        if not initialized:
            self.model.initialize(init=mx.init.Xavier(), ctx=self.ctx)
            self.model.encoder.weight.set_data(embed_weight)
        self.model.hybridize()

    def fit(self, train_dataset, valid_dataset=None,
            update_embedding=True, n_epochs=15,
            optimizer='adam', lr=3e-4, clip=5.0, verbose=True,
            checkpoint=None, save_frequency=1):
        """
        Fit the address tagger model.

        Parameters:
        ----
        train_dataset: list of tuples
          Each tuple is a (text, tags) pair.
        valid_dataset: list of tuples
          Each tuple is a (text, tags) pair. If None, valid log will be ignored
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
        if not update_embedding:
            self.model.encoder.weight.grad_req = 'null'
        dataset = []
        for text, tags in train_dataset:
            text = text[:self._max_length]
            # text = text.replace('&', ' ')
            inputs = self.vocab[list(char_cut_func(text))]
            tagidx = [self._tag2idx(tag)
                      for tag in tags.split('|')][:self._max_length]
            mask = np.ones(len(text), dtype=np.float32)
            dataset.append((inputs, tagidx, mask))

        train_batchify_fn = nlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1),
            Pad(axis=0),
            Pad(axis=0))

        train_dataloader = mx.gluon.data.DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            last_batch='discard',
            batchify_fn=train_batchify_fn,
            num_workers=4)

        self._fit(train_dataloader, lr, n_epochs,
                  valid_dataset=valid_dataset,
                  optimizer=optimizer, clip=clip, verbose=verbose,
                  checkpoint=checkpoint, save_frequency=save_frequency)

    def _valid_log(self, valid_dataset):
        self.decoder = self._create_decoder(
            self.model.crf_layer.transitions.data().asnumpy())
        scores = self.valid_score(valid_dataset[0], valid_dataset[1])
        self.logger.info(
            'province(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'city(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'district(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'location(ser: %.3f, cer: %.3f)\n' % tuple(scores))

    def _calculate_loss(self, batch_inputs, batch_tags, batch_mask):
        return self.model(batch_inputs.transpose(axes=(1, 0)),
                          batch_tags.transpose(axes=(1, 0)),
                          batch_mask.transpose(axes=(1, 0)))

    def predict(self, texts):
        dataset = []
        for text in texts:
            text = text[:self._max_length]
            # text = text.replace('&', ' ')
            inputs = self.vocab[list(self._char_cut_func(text))]
            mask = np.ones(len(text), dtype=np.float32)
            dataset.append((inputs, mask))

        batchify_fn = nlp.data.batchify.Tuple(
            Pad(axis=0, pad_val=1), Pad(axis=0))

        dataloader = mx.gluon.data.DataLoader(
            dataset=dataset,
            batch_size=512,
            shuffle=False,
            batchify_fn=batchify_fn)

        # list of lists
        # each list contains tag ids for one text
        data_tags = []
        for (batch_inputs, batch_inputs_mask) in dataloader:
            batch_inputs = batch_inputs.as_in_context(self.ctx)
            # batch_segments = batch_segments.as_in_context(self.ctx)
            batch_inputs_mask = batch_inputs_mask.as_in_context(self.ctx)

            emissions = self.encoder(batch_inputs.transpose(axes=(1, 0)))
            batch_pred = self.decoder(
                emissions, batch_inputs_mask.transpose(axes=(1, 0)))
            data_tags.extend(batch_pred)
        tags = list(itertools.chain(*[[self._idx2tag(i) for i in tags]
                                      for tags in data_tags]))
        return post_rule.TaggedAddress(''.join(texts), tags)

    def valid_score(self, texts, y, city_code='020', try_fix_tag=False):
        tagged_addresses = [self.predict(text.split('&')) for text in texts]
        for tagged_address in tagged_addresses:
            self._rule_set.apply(tagged_address, city_code=city_code,
                                 try_fix_tag=try_fix_tag)
        return score(y,
                     [[tagged_address.final_province or '',
                       tagged_address.final_city or '',
                       tagged_address.final_district or '',
                       tagged_address.final_location or '']
                      for tagged_address in tagged_addresses])

    def save_model(self, prefix):
        with open(f'{prefix}-vocab.json', 'w') as f:
            f.write(self.vocab.to_json())
        self.model.export(f'{prefix}', epoch=0)
        self.model.save_parameters(f'{prefix}-params')
        with open(f'{prefix}-meta.json', 'w') as f:
            f.write(json.dumps(self.meta))

    @classmethod
    def load_model(cls, tags, res,
                   vocab_file, meta_file, model_file):
        with open(vocab_file) as f:
            vocab = nlp.Vocab.from_json(f.read())
        with open(meta_file) as f:
            meta = json.loads(f.read())
        model = AddressTagger(**meta)
        model.load_parameters(model_file, ctx=try_gpu())
        return cls(vocab, tags, res,
                   max_length=meta['max_length'],
                   embed_size=meta['embed_size'],
                   hidden_size=meta['hidden_size'],
                   num_rnn_layers=meta['num_rnn_layers'],
                   activation=meta['activation'],
                   dropout=meta['dropout'],
                   model=model)
