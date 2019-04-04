from mxnet.gluon import nn, rnn


class BiLstmEncoder(nn.HybridBlock):

    def __init__(self, vocab_size, embed_size, num_tags, max_length=50,
                 hidden_size=512, num_rnn_layers=1, dropout=0.0,
                 activation='relu', prefix='encoder_'):
        super().__init__(prefix=prefix)
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._max_length = max_length
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(vocab_size, embed_size),
                init=mx.init.Uniform(0.1), grad_stype='row_sparse')
            self.embed_dropout = nn.Dropout(
                dropout, prefix='embed_dropout_')
            self.birnn_layer = rnn.LSTM(hidden_size // 2, num_rnn_layers,
                                        dropout=dropout, bidirectional=True,
                                        prefix='rnn_')
            self.rnn_dropout = nn.Dropout(
                dropout, prefix='rnn_dropout_')
            self.emission2tag_layer = nn.Dense(
                num_tags, activation=activation, flatten=False,
                prefix='emission2tag_')

    def hybrid_forward(self, F, inputs, weight):
        """
        inputs: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        # [seq_length, batch_size, embed_size]
        embed = self.embed_dropout(F.Embedding(inputs, weight,
                                               self._vocab_size,
                                               self._embed_size,
                                               sparse_grad=True))
        # embeds = F.concat(F.one_hot(seg_tags, 5), char_embeds, dim=-1)
        # [seq_length, batch_size, embed_size]
        rnn_outputs = self.rnn_dropout(self.birnn_layer(embed))
        return self.emission2tag_layer(rnn_outputs)
