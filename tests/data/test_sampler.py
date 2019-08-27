from sknlp.data.sampler import SequentialSampler, BucketSampler, BatchSampler


class TestSequentialSampler:

    def test_one_part(self):
        sampler = SequentialSampler(4)
        assert [i for i in sampler] == [0, 1, 2, 3]

    def test_multi_parts(self):
        sampler = SequentialSampler(4, num_parts=2, part_index=1)
        assert [i for i in sampler] == [2, 3]


class TestBucketSampler:

    def test_one_part(self):
        lengths = [1, 2, 3, 4, 5]
        # bucket key (3, 4, 5)
        # bucket info [(2, 0), (1, 0), (0, 0), (0, 2)]
        sampler = BucketSampler(lengths, 2, shuffle=False, num_buckets=3)
        assert [[i for i in idx] for idx in sampler] == [[4], [3], [0, 1], [2]]

        lengths = [1, 2, 3, 4, 5, 6]
        # bucket key (2, 4, 6)
        # bucket info [(2, 0), (1, 0), (0, 0)]
        sampler = BucketSampler(lengths, 2, shuffle=False, num_buckets=3)
        assert (
            [[i for i in idx] for idx in sampler] == [[4, 5], [2, 3], [0, 1]]
        )

    def test_multi_part(self):
        lengths = [1, 2, 3, 4, 5]
        sampler = BucketSampler(
            lengths, 2, shuffle=False, num_buckets=3, num_parts=2, part_index=1
        )
        assert [[i for i in idx] for idx in sampler] == [[3], [2]]

        lengths = [1, 2, 3, 4, 5, 6]
        sampler = BucketSampler(
            lengths, 2, shuffle=False, num_buckets=3, num_parts=2, part_index=1
        )
        assert (
            [[i for i in idx] for idx in sampler] == [[2, 3]]
        )


class TestBatchSampler:

    data = list(range(10))

    def test_batch_size(self):
        sampler = BatchSampler(self.data, 3, sampler='sequential')
        assert sampler.batch_axis == 1
        assert sampler.batch_size == 3
        assert(
            [batch for batch in sampler] ==
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        )

        sampler = BatchSampler(self.data, 5, sampler='sequential')
        assert sampler.batch_size == 5
        assert (
            [batch for batch in sampler] ==
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        )

    def test_last_batch(self):
        sampler = BatchSampler(
            self.data, 3, sampler='sequential', last_batch='rollover'
        )
        assert(
            [batch for batch in sampler] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        )
        assert(
            [batch for batch in sampler] == [[9, 0, 1], [2, 3, 4], [5, 6, 7]]
        )
