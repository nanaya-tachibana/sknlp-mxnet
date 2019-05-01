import logging

import mxnet as mx


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)


class BaseModel:

    def __init__(self, **kwargs):
        self.logger = logger

    def _fit(self, train_dataloader: mx.gluon.data.DataLoader,
             valid_dataloader: mx.gluon.data.DataLoader = None,
             lr: float = 0.01,
             n_epochs: int = 10,
             optimizer: str = 'adam',
             update_steps_lr: int = 500,
             factor: float = 1,
             stop_factor_lr: float = 2e-6,
             clip: float = 5,
             verbose: bool = True,
             checkpoint: str = None,
             save_frequency: int = 1):
        """
        Help function for model fitting.

        Parameters:
        ----------
        train_dataloader: list of tuples
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
        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            update_steps_lr, factor=factor, stop_factor_lr=stop_factor_lr)

        assert (getattr(self, '_trainable', None) is not None and
                len(self._trainable) > 0), 'No trainable parameters'

        params_dict = self._get_trainable_params()
        trainer = mx.gluon.Trainer(params_dict,
                                   optimizer,
                                   {'learning_rate': lr,
                                    'lr_scheduler': lr_scheduler})
        for epoch in range(1, n_epochs + 1):
            self._one_epoch(trainer, train_dataloader, epoch, clip=clip)
            self._trained = True
            if checkpoint is not None and epoch % save_frequency == 0:
                self.save_model(f'{checkpoint}')
            # valid
            if valid_dataset is not None:
                self._valid_log(valid_dataset)

    def _get_trainable_params(self):
        params_dict = self._trainable[0].collect_params()
        for t in self._trainable[1:]:
            params_dict.update(t.collect_params())
        return params_dict

    def _clip_gradient(self, clip):
        params_dict = self._get_trainable_params()
        clip_params = [
            p.data() for p in params_dict.values()]

        norm = mx.nd.array([0.0], self._ctx)
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
        ctx = self._ctx
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


class DeepModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_net(self):
        """
        Implement this function to build net.
        """
        raise NotImplementedError('build net is not implemented.')

    def _build(self):
        """
        Implement this function to build.
        """
        raise NotImplementedError('build is not implemented.')

    def _initialize_net(self):
        self._net.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        self._loss.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        if self._embed_weight is not None:
            self._net.embedding_layer.weight.set_data(self._embed_weight)
        self._net.hybridize()
        self._loss.hybridize()

    def _build_dataloader(self, dataset, batch_size, shuffle, last_batch):
        return mx.gluon.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        last_batch=last_batch,
                                        batchify_fn=self._batchify_fn)

    def _get_or_build_dataset(self, dataset, X, y):
        """
        Implement this function to build net.
        """
        raise NotImplementedError('build net is not implemented.')

    def fit(self, X=None, y=None, train_dataset=None,
            valid_X=None, valid_y=None, valid_dataset=None,
            batch_size=32, last_batch='keep',
            update_embedding=True, n_epochs=15,
            optimizer='adam', lr=3e-4, clip=5.0, verbose=True,
            checkpoint=None, save_frequency=1):
        """
        Fit model.

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
        train_dataset = self._get_or_build_dataset(train_dataset, X, y)
        assert self._num_classes == len(train_dataset.label2idx)

        self.idx2labels = train_dataset.idx2labels
        if self._vocab is None:
            self._vocab = train_dataset.vocab
            self._build()
        if self._label2idx is None:
            self._label2idx = train_dataset.label2idx
            self.meta['label2idx'] = self._label2idx

        if valid_X and valid_y and valid_dataset is None:
            valid_dataset = self._get_or_build_dataset(train_dataset, X, y)

        if not update_embedding:
            self._net.embedding_layer.weight.grad_req = 'null'

        dataloader = self._build_dataloader(train_dataset, batch_size,
                                            True, last_batch)
        self._fit(dataloader, lr, n_epochs,
                  valid_dataset=valid_dataset,
                  optimizer=optimizer, clip=clip, verbose=verbose,
                  checkpoint=checkpoint, save_frequency=save_frequency)
