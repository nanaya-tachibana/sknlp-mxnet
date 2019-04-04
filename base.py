import logging

import mxnet as mx
import gluonnlp as nlp
from gensim.models import KeyedVectors


class DeepModel:

    def __init__(self, vocab, encoder, loss, ctx, log_file='train.log'):
        self.ctx = ctx
        self.vocab = vocab
        self.encoder = loss
        self.loss = loss

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.WARNING)
        self.stream_log = logging.StreamHandler()
        self.stream_log.setLevel(level=logging.WARNING)
        self.file_log = logging.FileHandler(log_file)
        self.file_log.setLevel(level=logging.WARNING)
        self.logger.addHandler(self.stream_log)
        self.logger.addHandler(self.file_log)

    def _fit(self, train_dataloader, lr, n_epochs,
             valid_dataset=None,
             optimizer='adam', update_steps_lr=500,
             factor=1, stop_factor_lr=2e-6, clip=5,
             verbose=True, checkpoint=None, save_frequency=1):
        """
        Help function for model fitting.

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
        lr_update_steps : int
          Changes the learning rate for every n updates.
        factor : float
          The factor to change the learning rate.
        stop_factor_lr : float
          Stop updating the learning rate if it is less than this value.
        verbose:
          If true, training loss and validation score will be logged.
        checkpoint: str
          If not None, save model using `checkpoint` as prefix.
        save_frequency: int
          If checkpoint is not None, save model every `save_frequency` epochs.
        """
        if verbose:
            self.logger.setLevel(level=logging.INFO)
            self.stream_log.setLevel(level=logging.INFO)
            self.file_log.setLevel(level=logging.INFO)
        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            update_steps_lr, factor=factor, stop_factor_lr=stop_factor_lr)
        params_dict = self.encoder.collect_params()
        params_dict.update(self.loss.collect_params())
        trainer = mx.gluon.Trainer(params_dict,
                                   optimizer,
                                   {'learning_rate': lr,
                                    'lr_scheduler': lr_scheduler})
        for epoch in range(1, n_epochs + 1):
            self._one_epoch(trainer, train_dataloader, epoch, clip=clip)
            if checkpoint is not None and epoch % save_frequency == 0:
                if epoch == 1:
                    # if hasattr(self.vocab, 'embedding'):
                    #     self.vocab.embedding.serialize(f'{checkpoint}-vocab')
                    with open(f'{checkpoint}-vocab.json', 'w') as f:
                        f.write(self.vocab.to_json())
                self.model.export(f'{checkpoint}', epoch=epoch)
                self.model.save_parameters(f'{checkpoint}-{epoch:04}-params')
            # valid
            if valid_dataset is not None:
                self._valid_log(valid_dataset)

    def _clip_gradient(self, clip):
        params_dict = self.encoder.collect_params()
        params_dict.update(self.loss.collect_params())
        clip_params = [
            p.data() for p in params_dict.values()]

        norm = mx.nd.array([0.0], self.ctx)
        for param in clip_params:
            if param.grad is not None:
                norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > clip:
            for param in clip_params:
                if param.grad is not None:
                    param.grad[:] *= clip / norm

    def _one_epoch(self, trainer, data_iter, epoch, clip=5.0):
        loss_val = 0.0
        n_batch = 0
        ctx = self.ctx
        for one_batch in data_iter:
            one_batch = [element.as_in_context(ctx) for element in one_batch]
            steps = one_batch[0].shape[0]

            with mx.autograd.record():
                loss = self._calculate_loss(*one_batch)
            loss.backward()
            self._clip_gradient(clip)
            trainer.step(steps, ignore_stale_grad=True)
            batch_loss = loss.mean().asscalar()
            n_batch += 1
            loss_val += batch_loss
            # check the result of traing phase
            if n_batch % 10 == 0:
                self.logger.info(f'epoch {epoch}, batch {n_batch}, '
                                 f'batch_train_loss: {batch_loss:.4}')
        return loss_val / n_batch

    def _calculate_loss(self, *args):
        """
        Implement this function to calculate the loss.

        Parameters:
        args: list
          A list of args in data_iter.
          The order of args is the same as the order in train dataset.
        ----
        """
        raise NotImplementedError('loss function is not implemented.')

    def _valid_log(self, valid_dataset):
        """
        Implement this function to calculate the loss.

        Parameters:
        valid_dataset: valid dataset given in _fit function.
        ----
        """
        raise NotImplementedError('valid log function is not implemented.')

    def save_model(self, file_preifx=''):
        pass

    @classmethod
    def create_vocab_from_word2vec_file(cls, word2vec_file, binary=True):
        embed = KeyedVectors.load_word2vec_format(word2vec_file, binary=binary)
        counter = nlp.data.count_tokens(embed.vocab.keys())
        vocab = nlp.Vocab(counter)

        embed_weight = embed[vocab.idx_to_token[4:]]
        embed_weight = mx.nd.concat(mx.nd.zeros((4, embed_weight.shape[1])),
                                    mx.nd.array(embed_weight),
                                    dim=0)
        return vocab, embed_weight
