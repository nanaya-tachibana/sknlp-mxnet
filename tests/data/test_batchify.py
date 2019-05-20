from sknlp.data import BPTTBatchify


def test_bpttbatchify():
    batchify = BPTTBatchify(1, 2, 3)
    data = [[8, 9, 10], [100, 200, 300, 400, 500]]
    tokens, target_tokens, reverse_target_tokens = batchify(data)
    assert tokens.asnumpy().tolist() == [
        [8, 9, 10, 3, 1, 1],
        [100, 200, 300, 400, 500, 3]
    ]
    assert target_tokens.asnumpy().tolist() == [
        [9, 10, 3, 1, 1, 1],
        [200, 300, 400, 500, 3, 1]
    ]
    assert reverse_target_tokens.asnumpy().tolist() == [
        [2, 8, 9, 10, 1, 1],
        [2, 100, 200, 300, 400, 500]
    ]
