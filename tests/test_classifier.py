import mxnet as mx
from sknlp.classifier import DeepClassifier
from sknlp.vocab import Vocab


class TestDeepClassifier:

    vocab = Vocab()
    clf = DeepClassifier(2, None, vocab=vocab, ctx=mx.cpu())

    def test_batchify_fn(self):
        batchify = self.clf._batchify_fn()
        data = [
            ([8, 9, 10], 0),
            ([100, 200, 300, 400, 500], 1)
        ]
        batch_inputs, batch_mask, batch_labels = batchify(data)
        assert batch_inputs.asnumpy().transpose().tolist() == [
            [8, 9, 10, 1, 1],
            [100, 200, 300, 400, 500]
        ]
        assert batch_mask.asnumpy().transpose().tolist() == [
            [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]
        ]
        assert batch_labels.asnumpy().transpose().tolist() == [0, 1]
