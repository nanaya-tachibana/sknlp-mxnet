import random
import numpy as np
import mxnet as mx
from mxnet.gluon.data.sampler import RandomSampler, SequentialSampler

from gluonnlp.data.stream import _ProcessPrefetcher


class BPTTBatchSampler:

    def __init__(
        self, dataset, batch_size, seq_len, eos_token, padding_token,
        sampler='random', last_batch='keep', num_prefetch=1
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._eos_token = eos_token
        self._padding_token = padding_token
        self._sampler = self._get_sampler(sampler)(len(self._dataset))
        self._last_batch = last_batch
        self._num_prefetch = num_prefetch

    def _get_sampler(self, sampler):
        assert isinstance(
            sampler, str
        ), 'Expected sampler to be a str, but got %s' % type(sampler)
        if sampler == 'random':
            return RandomSampler
        if sampler == 'sequential':
            return SequentialSampler
        raise ValueError(
            'sampler must be either "random" or "sequential", but got %s' %
            (sampler)
        )

    def __iter__(self):
        seed = random.getrandbits(32)
        np_seed = np.random.randint(0, 2**32)
        mx_seed = int(mx.nd.random.uniform(0, 2**32).asscalar())
        corpus = (self._dataset[idx] for idx in self._sampler)

        def _init():
            return (
                np.full(
                    [self._batch_size, self._seq_len],
                    self._padding_token, dtype=np.float32
                ),
                np.zeros(
                    [self._batch_size, self._seq_len], dtype=np.float32
                ),
                np.full(
                    [self._batch_size, self._seq_len],
                    self._padding_token, dtype=np.float32
                ),
                np.full(
                    [self._batch_size, self._seq_len],
                    self._padding_token, dtype=np.float32
                )
            )

        def _read(buffers, i, corpus):
            """Read a sentence from the corpus into i-th buffer."""
            if len(buffers[i]) <= 2:
                buffers[i].extend(next(corpus) + [self._eos_token])

        def _write(
            data, mask, target, reverse_target, buffers, seq_len, i, length
        ):
            """Write a sentence from i-th buffer to data and target."""
            num_tokens = len(buffers[i]) - 2
            num_tokens = min(num_tokens, seq_len - length)
            # fill in data and target
            start_idx, end_idx = length, length + num_tokens
            data[i, start_idx:end_idx] = buffers[i][1:num_tokens + 1]
            target[i, start_idx:end_idx] = buffers[i][2:num_tokens + 2]
            reverse_target[i, start_idx:end_idx] = buffers[i][:num_tokens]
            mask[i, start_idx:end_idx] = 1
            # trim sentence in the buffer if too long. Used for the next batch
            buffers[i] = buffers[i][num_tokens:]
            return num_tokens

        # stream states
        buffers = [[self._eos_token] for _ in range(self._batch_size)]
        has_next = True
        has_token_buffered = False
        while has_next or has_token_buffered > 0:
            data, mask, target, reverse_target = _init()
            has_token_buffered = False
            for i in range(self._batch_size):
                length = 0
                try:
                    while length < self._seq_len:
                        _read(buffers, i, corpus)
                        num_tokens = _write(
                            data, mask, target, reverse_target, buffers,
                            self._seq_len, i, length
                        )
                        if len(buffers[i]) > 2:
                            has_token_buffered = True
                        length += num_tokens
                except StopIteration:
                    has_next = False
            boolean_idx = mask.sum(axis=1) != 0
            num_batch = sum(boolean_idx)
            if num_batch == self._batch_size or self._last_batch == 'keep':
                yield (
                    mx.nd.array(data).T, mx.nd.array(mask).T,
                    mx.nd.array(target).T, mx.nd.array(reverse_target).T
                )
