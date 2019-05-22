import logging
import mxnet as mx
from .data import NLPDataset

logger = logging.getLogger(__name__)


class BaseModel:

    def __init__(self, ctx):
        self._ctx = ctx
        self._trainable = dict()

    def _fit(self, train_dataloader: mx.gluon.data.DataLoader,
             valid_dataset: NLPDataset,
             lr: float = 0.01,
             n_epochs: int = 10,
             optimizer: str = 'adam',
             update_steps_lr: int = 300,
             factor: float = 0.9,
             stop_factor_lr: float = 2e-6,
             clip: float = 5,
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
        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            update_steps_lr, factor=factor,
            stop_factor_lr=stop_factor_lr
        )

        assert len(self._trainable) > 0, 'No trainable parameters'

        params_dict = self._collect_params()
        trainer = mx.gluon.Trainer(params_dict,
                                   optimizer,
                                   {'learning_rate': lr,
                                    'lr_scheduler': lr_scheduler})
        for epoch in range(1, n_epochs + 1):
            self._one_epoch(trainer, train_dataloader, epoch, clip=clip)
            self._trained = True
            if checkpoint is not None and epoch % save_frequency == 0:
                self.save(f'{checkpoint}-{epoch:04}')

            if valid_dataset is not None:
                self._valid_log(valid_dataset)

    def _collect_params(self):
        params_dict = mx.gluon.ParameterDict()
        for t in self._trainable:
            params_dict.update(self._trainable[t].collect_params())
        return params_dict

    def _clip_gradient(self, clip):
        params_dict = self._collect_params()
        clip_params = [
            p.data() for p in params_dict.values()
        ]

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
            if n_batch % 100 == 0:
                logger.info(f'epoch {epoch}, batch {n_batch}, '
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

    def _batchify_fn(self):
        raise NotImplementedError('_batchify_fn is not implemented.')

    def _build_dataloader(
        self, dataset, batch_size, shuffle=True, last_batch='keep'
    ):
        return mx.gluon.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        last_batch=last_batch,
                                        batchify_fn=self._batchify_fn())

    def save(self, file_path: str) -> None:
        raise NotImplementedError('save model function is not implemented.')

    @classmethod
    def load(cls, file_path, ctx=mx.cpu()):
        raise NotImplementedError('load model function is not implemented.')


class DeepSupervisedModel(BaseModel):

    def __init__(self, vocab=None, label2idx=None, ctx=mx.cpu(), **kwargs):
        super().__init__(ctx, **kwargs)
        self._vocab = vocab
        self._label2idx = label2idx
        self._loss = None
        self.meta = dict()

    def _build(self):
        """
        Implement this function to build.
        """
        raise NotImplementedError('build is not implemented.')

    def _initialize(self):
        self.embedding_layer.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        self.encode_layer.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        self._loss.initialize(init=mx.init.Xavier(), ctx=self._ctx)
        self.embedding_layer.hybridize()
        self.encode_layer.hybridize()
        self._loss.hybridize()

    def _get_or_build_dataset(self, dataset, X, y):
        """
        Implement this function to build dataset.
        """
        raise NotImplementedError('build dataset is not implemented.')

    def fit(
        self, X=None, y=None, train_dataset=None,
        valid_X=None, valid_y=None, valid_dataset=None, batch_size=32,
        last_batch='keep', n_epochs=15, optimizer='adam', lr=3e-4,
        update_steps_lr: int = 300, factor: float = 0.9,
        stop_factor_lr: float = 2e-6, clip=5.0, checkpoint=None,
        save_frequency=1
    ):
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

        self.idx2labels = train_dataset.idx2labels
        self.label_weights = mx.nd.array(
            train_dataset._label_weights, ctx=self._ctx
        )

        if self._vocab is None:
            self._vocab = train_dataset._vocab
        if self._label2idx is None:
            self._label2idx = train_dataset._label2idx
            self.meta['label2idx'] = self._label2idx
        if not self._trained:
            self._build()
            self._initialize()

        if valid_X and valid_y and valid_dataset is None:
            valid_dataset = self._get_or_build_dataset(
                valid_dataset, valid_X, valid_y
            )

        dataloader = self._build_dataloader(
            train_dataset, batch_size, shuffle=True, last_batch=last_batch
        )
        self._fit(
            dataloader, valid_dataset, lr=lr, n_epochs=n_epochs,
            update_steps_lr=update_steps_lr, factor=factor,
            stop_factor_lr=stop_factor_lr, optimizer=optimizer, clip=clip,
            checkpoint=checkpoint, save_frequency=save_frequency
        )
