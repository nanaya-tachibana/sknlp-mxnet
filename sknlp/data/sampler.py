import itertools

import numpy as np
from gluonnlp.data.sampler import SplitSampler as RandomSampler
from gluonnlp.data.sampler import FixedBucketSampler


class SequentialSampler(RandomSampler):

    def __iter__(self):
        return iter(range(self._start, self._end))


class BucketSampler(FixedBucketSampler):

    def __init__(
        self, lengths, batch_size, num_buckets=10, shuffle=True,
        num_parts=1, part_index=0
    ):
        super().__init__(
            lengths, batch_size, num_buckets=num_buckets, shuffle=shuffle,
            num_shards=0 if num_parts == 1 else num_parts
        )
        self._part_index = part_index

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._batch_infos)
            for bucket_id in range(len(self._bucket_keys)):
                np.random.shuffle(self._bucket_sample_ids[bucket_id])

        if self._num_shards > 0:
            num_batches = len(self._batch_infos)
            for batch_idx in range(0, num_batches, self._num_shards):
                if batch_idx + self._part_index >= len(self._batch_infos):
                    return
                batch_info = self._batch_infos[batch_idx + self._part_index]
                bucket_id, batch_begin = batch_info
                batch_size = self._bucket_batch_sizes[bucket_id]
                batch_end = min(
                    batch_begin + batch_size,
                    len(self._bucket_sample_ids[bucket_id])
                )
                yield self._bucket_sample_ids[bucket_id][batch_begin:batch_end]
        else:
            for bucket_id, batch_begin in self._batch_infos:
                batch_size = self._bucket_batch_sizes[bucket_id]
                batch_end = min(
                    batch_begin + batch_size,
                    len(self._bucket_sample_ids[bucket_id])
                )
                yield self._bucket_sample_ids[bucket_id][batch_begin:batch_end]


class BatchSampler:

    def __init__(
        self, dataset, batch_size, batch_axis=1, sampler='random',
        last_batch='keep', batchify_fn=None, num_parts=1, part_index=0,
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._batch_axis = batch_axis
        self._num_parts = num_parts
        self._part_index = part_index
        self._sampler = self._get_sampler(sampler)
        self._last_batch = last_batch
        self._batchify_fn = batchify_fn
        self._prev = []

    def _get_sampler(self, sampler):
        assert isinstance(
            sampler, str
        ), 'Expected sampler to be a str, but got %s' % type(sampler)
        if sampler == 'random':
            return RandomSampler(
                len(self._dataset),
                num_parts=self._num_parts, part_index=self._part_index
            )
        if sampler == 'sequential':
            return SequentialSampler(
                len(self._dataset),
                num_parts=self._num_parts, part_index=self._part_index
            )
        if sampler == 'bucket':
            assert hasattr(
                self._dataset, 'text_lengths',
            ), 'When use bucket sampler, dataset must have '
            'text_lengths property which returns a list of lengths of samples.'
            return BucketSampler(
                self._dataset.text_lengths, self._batch_size,
                num_parts=self._num_parts, part_index=self._part_index
            )
        raise ValueError(
            'sampler must be either "random" or "sequential", but got %s' %
            (sampler)
        )

    def __iter__(self):
        if isinstance(self._sampler, BucketSampler):
            corpus = itertools.chain.from_iterable(
                (self._dataset[idx] for idx in batch_idx)
                for batch_idx in self._sampler
            )
        else:
            corpus = (self._dataset[idx] for idx in self._sampler)
        batch, self._prev = self._prev, []
        for i in corpus:
            batch.append(i)
            if len(batch) == self._batch_size:
                if callable(self._batchify_fn):
                    yield self._batchify_fn(batch)
                else:
                    yield batch
                batch = []
        if batch:
            if self._last_batch == 'keep':
                if callable(self._batchify_fn):
                    yield self._batchify_fn(batch)
                else:
                    yield batch
            elif self._last_batch == 'discard':
                return
            elif self._last_batch == 'rollover':
                self._prev = batch
            else:
                raise ValueError(
                    "last_batch must be one of 'keep', "
                    "'discard', or 'rollover', but got %s" % self._last_batch
                )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_axis(self):
        return self._batch_axis


class BPTTBatchSampler(BatchSampler):

    def __init__(
        self, dataset, batch_size, seq_len,
        bos_token, eos_token, padding_token,
        sampler='random', last_batch='keep',
        num_parts=1, part_index=0
    ):
        super().__init__(
            dataset, batch_size, sampler=sampler, last_batch=last_batch,
            num_parts=num_parts, part_index=part_index
        )
        self._seq_len = seq_len
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._padding_token = padding_token

    def __iter__(self):
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
                buffers[i].extend(
                    [self._bos_token] + next(corpus) + [self._eos_token]
                )

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
                yield data.T, mask.T, target.T, reverse_target.T
