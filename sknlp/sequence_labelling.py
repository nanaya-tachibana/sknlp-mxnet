import mxnet as mx
from mxnet.gluon import nn

from .base import DeepModel
from .encoder import BiLstmEncoder
from .crf import Crf
from .utils import try_gpu


class AddressTagger(nn.HybridBlock):

    def __init__(self, vocab_size, embed_size, num_tags, max_length=50,
                 hidden_size=512, num_rnn_layers=1, dropout=0.0,
                 activation='relu', prefix='tagger_'):
        super().__init__(prefix=prefix)
        with self.name_scope():
            self.encoder = BiLstmEncoder(vocab_size, embed_size, num_tags,
                                         max_length=50, hidden_size=512,
                                         num_rnn_layers=1, dropout=0.0,
                                         activation='relu')
            self.crf_layer = Crf(num_tags)

    def hybrid_forward(self, F, inputs, tags, mask):
        """
        inputs: shape(seq_length, batch_size)
        tags: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        # [seq_length, batch_size, num_tags]
        emissions = self.encoder(inputs)
        return -self.crf_layer(emissions, tags, mask)


class LstmSequenceLabel(DeepModel):

    def __init__(self, vocab, tags, decoder='crf',
                 ctx=mx.cpu(), max_length=50,
                 embed_size=300, hidden_size=512,
                 num_rnn_layers=1, activation='relu', dropout=0.0,
                 model=None, embed_weight=None):
        self.tag2idx = dict(zip(tags, range(len(tags))))
        self.idx2tag = dict(zip(self.tag2idx.values(), self.tag2idx.keys()))

        self.meta = {
            'vocab_size': len(vocab),
            'embed_size': embed_size,
            'num_tags': len(self.tag2idx),
            'max_length': max_length,
            'hidden_size': hidden_size,
            'num_rnn_layers': num_rnn_layers,
            'activation': activation,
            'dropout': dropout
        }
        self._max_length = max_length
        initialized = False if model is None else True
        if model is None:
            model = AddressTagger(**self.meta)
        super().__init__(vocab, model, ctx)
        self.encoder = self.model.encoder
        if not initialized:
            self.model.initialize(init=mx.init.Xavier(), ctx=ctx)
            self.model.encoder.weight.set_data(embed_weight)
        self.model.hybridize()

    def _tag2idx(self, tag):
        return self.tag2idx.get(tag, 0)

    def _idx2tag(self, idx):
        return self.idx2tag.get(idx, 'O')

    def _idx2tokens(self, indices, mask_list=None):
        """
        Convert a list of texts from idx representation into origin texts.

        Parameters:
        ----
        indices: list of lists
          Each list contains token ids for one text.
        mask_list: list of lists
          Each list contains a 01 mask.
        """
        tokens = []
        indices = [[int(i) for i in idx] for idx in indices]
        if mask_list is None:
            for idx in indices:
                token = self.vocab.to_tokens(idx)
                for i, t in enumerate(token):
                    if t == '<unk>':
                        token[i] = ' '
                tokens.append(''.join(token))
        else:
            for idx, mask in zip(indices, mask_list):
                length = int(sum(mask))
                idx = idx[:length]
                token = self.vocab.to_tokens(idx)
                for i, t in enumerate(token):
                    if t == '<unk>':
                        token[i] = ' '
                tokens.append(''.join(token))
        return tokens

    def fit(self, train_dataset, valid_dataset=None,
            char_cut_func=char_cut_func,
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
        cut_func: function
          Function used to segment text.
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
        self._char_cut_func = char_cut_func
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
            batchify_fn=train_batchify_fn)

        self._fit(train_dataloader, lr, n_epochs,
                  valid_dataset=valid_dataset,
                  optimizer=optimizer, clip=clip, verbose=verbose,
                  checkpoint=checkpoint, save_frequency=save_frequency)

    def _valid_log(self, valid_dataset):
        self.decoder = self._create_decoder(
            self.model.crf_layer.transitions.data().asnumpy())
        scores = self.valid_score(valid_dataset[0],
                                  valid_dataset[1],
                                  valid_dataset[2])
        self.logger.info(
            'province(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'city(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'district(precision: %.3f, recall: %.3f, f1: %.3f)\n'
            'location(ser: %.3f, cer: %.3f)\n' % tuple(scores))

    def _calculate_loss(self, batch_inputs, batch_tags, batch_mask):
        return self.model(batch_inputs.transpose(axes=(1, 0)),
                          batch_tags.transpose(axes=(1, 0)),
                          batch_mask.transpose(axes=(1, 0)))

    def _create_decoder(self, transitions):

        def decoder(inputs, mask=None):
            vd = functools.partial(viterbi_decode, transitions)
            return vd(inputs.asnumpy(), mask=mask.asnumpy())

        return decoder

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

    def valid_score(self, texts, y, city_codes, try_fix_tag=False):
        tagged_addresses = [self.predict(text.split('&')) for text in texts]
        for tagged_address, city_code in zip(tagged_addresses, city_codes):
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
