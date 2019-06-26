import time
import logging
import mxnet as mx

from .data import NLPDataset
from .data.sampler import BatchSampler
from .data.dataloader import PrefetchDataLoader

logger = logging.getLogger(__name__)


class BaseModel:

    def __init__(self, ctx=None):
        self._ctx = [mx.cpu()] if ctx is None else ctx
        self._prefetch = 0
        self._trained = False
        self._trainable = dict()

    def _fit(
        self, train_dataloader,
        valid_dataset: NLPDataset,
        lr: float = 0.001,
        n_epochs: int = 100,
        optimizer: str = 'adam',
        lr_update_factor: float = 0.9,
        lr_update_steps: int = 1000,
        clip: float = 5,
        checkpoint: str = None,
        save_frequency: int = 1,
    ):
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
        assert len(self._trainable) > 0, 'No trainable parameters'

        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            step=lr_update_steps, factor=lr_update_factor
        )
        optimizer_params = {'learning_rate': lr, 'lr_scheduler': lr_scheduler}
        params_dict = self._collect_params()
        trainer = mx.gluon.Trainer(params_dict, optimizer, optimizer_params)
        for epoch in range(1, n_epochs + 1):
            self._before_epoch(trainer=trainer, dataloader=train_dataloader)
            avg_loss = self._one_epoch(trainer, train_dataloader, epoch, clip)
            self._train_log(avg_loss)
            self._trained = True
            if checkpoint is not None and epoch % save_frequency == 0:
                self.save(f'{checkpoint}-{epoch:04}')

            if valid_dataset is not None:
                score = self._valid_log(valid_dataset)

    def _collect_params(self):
        params_dict = mx.gluon.ParameterDict()
        for t in self._trainable:
            params_dict.update(self._trainable[t].collect_params())
        return params_dict

    def _clip_gradient(self, clip, ctx):
        params_dict = self._collect_params()
        for context in ctx:
            clip_params = [
                p.data(context) for p in params_dict.values()
            ]
            norm = mx.nd.array([0.0], context)
            for param in clip_params:
                if param.grad is not None:
                    norm += (param.grad ** 2).sum()
            norm = norm.sqrt().asscalar()
            if norm > clip:
                for param in clip_params:
                    if param.grad is not None:
                        param.grad[:] *= clip / norm

    def _forward(self, func, one_batch, ctx, batch_axis=1):
        res = []
        for one_part in zip(*[
            mx.gluon.utils.split_and_load(
                element, ctx, batch_axis=batch_axis
            ) for element in one_batch
        ]):
            res.append(func(*one_part))
        return res

    def _forward_backward(self, one_batch, ctx, batch_axis=1):
        with mx.autograd.record():
            res = self._forward(
                self._calculate_loss, one_batch, ctx, batch_axis
            )
            losses = [r[0] for r in res]
        for loss in losses:
            loss.backward()
        return sum(loss.sum().asscalar() for loss in losses)

    def _before_epoch(self, *arg, **kwargs):
        pass

    def _one_epoch(self, trainer, data_iter, epoch, clip=1.0):
        total_loss = 0
        num_batch = 0
        ctx = self._ctx
        start_time = time.time()
        batch_axis = data_iter._batch_axis
        for one_batch in data_iter:
            steps = one_batch[0].shape[batch_axis]
            loss = self._forward_backward(one_batch, ctx, batch_axis)
            self._clip_gradient(clip, ctx)
            trainer.step(steps, ignore_stale_grad=True)
            batch_loss = self._batch_loss(loss, *one_batch)
            total_loss += batch_loss
            num_batch += 1
            if num_batch % 100 == 0:
                speed = round(num_batch / (time.time() - start_time), 2)
                logger.info(
                    f'epoch {epoch}, batch {num_batch}, '
                    f'batch_train_loss: {batch_loss:.4}, '
                    f'speed: {speed} batches/s.'
                )
        return total_loss / num_batch

    def _batch_loss(self, loss, *args):
        return loss / args[0].shape[1]

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

    def _train_log(self, loss):
        logger.info(f'train loss: {loss}')

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
        batch_sampler = BatchSampler(
            dataset, batch_size,
            sampler='random' if shuffle else 'sequential',
            last_batch=last_batch, _batchify_fn=self._batchify_fn()
        )
        if self._prefetch > 0:
            return PrefetchDataLoader(batch_sampler, batch_size)
        else:
            return batch_sampler

    def save(self, file_path: str) -> None:
        raise NotImplementedError('save model function is not implemented.')

    @classmethod
    def load(cls, file_path, ctx=mx.cpu()):
        raise NotImplementedError('load model function is not implemented.')


class DeepSupervisedModel(BaseModel):

    def __init__(self, vocab=None, label2idx=None, ctx=None, **kwargs):
        super().__init__(ctx, **kwargs)
        self._vocab = vocab
        self._label2idx = label2idx
        self._loss = None
        self.meta = dict()

    def _build(self, ctx):
        """
        Implement this function to build.
        """
        raise NotImplementedError('build is not implemented.')

    def _get_or_build_dataset(self, dataset, X, y):
        """
        Implement this function to build dataset.
        """
        raise NotImplementedError('build dataset is not implemented.')

    def fit(
        self, X=None, y=None, train_dataset=None,
        valid_X=None, valid_y=None, valid_dataset=None, batch_size=32,
        last_batch='keep', n_epochs=15, optimizer='adam', lr=1e-3,
        lr_update_factor: float = 0.9,
        lr_update_steps: int = 1000,
        clip=5.0, checkpoint=None, save_frequency=1, prefetch=0
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
        self._prefetch = prefetch
        train_dataset = self._get_or_build_dataset(train_dataset, X, y)

        self.idx2labels = train_dataset.idx2labels
        if self._vocab is None:
            self._vocab = train_dataset._vocab
        if self._label2idx is None:
            self._label2idx = train_dataset._label2idx
            self.meta['label2idx'] = self._label2idx
        if not self._trained:
            self._build(self._ctx)

        if valid_X and valid_y and valid_dataset is None:
            valid_dataset = self._get_or_build_dataset(
                valid_dataset, valid_X, valid_y
            )

        dataloader = self._build_dataloader(
            train_dataset, batch_size, shuffle=True, last_batch=last_batch
        )
        self._fit(
            dataloader, valid_dataset, lr=lr, n_epochs=n_epochs,
            optimizer=optimizer, lr_update_factor=lr_update_factor,
            lr_update_steps=lr_update_steps, clip=clip, checkpoint=checkpoint,
            save_frequency=save_frequency
        )
