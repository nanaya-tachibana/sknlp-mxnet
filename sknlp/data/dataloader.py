import random

import numpy as np
import mxnet as mx
from gluonnlp.data.stream import _ProcessPrefetcher


class DataLoader:

    def __init__(self, batch_sampler, batch_size, batch_axis=1):
        self._batch_sampler = batch_sampler
        self._batch_size = batch_size
        self._batch_axis = batch_axis

    def __iter__(self):
        return iter(self._batch_sampler)

    def reset(self):
        pass


class PrefetchDataLoader(DataLoader):

    def __iter__(self):
        seed = random.getrandbits(32)
        np_seed = np.random.randint(0, 2**32)
        mx_seed = int(mx.nd.random.uniform(0, 2**32).asscalar())
        self._fetcher = _ProcessPrefetcher(
            self._batch_sampler, 1, seed=seed, np_seed=np_seed, mx_seed=mx_seed
        )
        return self._fetcher

    def reset(self):
        if self._fetcher:
            self._fetcher.terminate()
            self._fetcher = None
