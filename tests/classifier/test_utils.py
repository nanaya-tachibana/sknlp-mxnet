import numpy as np

from sknlp.classifier import logits2classes


class TestLogits2classes:

    def test_multilabel_input(self):
        logits = np.array([
            [1, 1, -1],
            [10, -1, 10],
            [-1, -1, -1]
        ])
        classes = logits2classes(logits, True)
        assert classes == [[0, 1], [0, 2], []]

    def test_multilabel_input_with_large_threshold(self):
        logits = np.array([
            [1, 1, -1],
            [10, -1, 10],
            [-1, -1, -1]
        ])
        classes = logits2classes(logits, True, threshold=0.9)
        assert classes == [[], [0, 2], []]

    def test_multilabel_input_with_threshold_vector(self):
        logits = np.array([
            [1, 1, -1],
            [10, -1, 10],
            [-1, -1, -1]
        ])
        classes = logits2classes(
            logits, True, threshold=np.array([0.5, 0.9, 0.2])
        )
        assert classes == [[0, 2], [0, 2], [2]]

    def test_binary_input(self):
        logits = np.array([
            [1, 2],
            [2, -1],
            [-1, -10]
        ])
        classes = logits2classes(logits, False)
        assert classes == [1, 0, 0]

    def test_binary_input_with_small_threshold(self):
        logits = np.array([
            [1, 2],
            [2, -1],
            [-1, -10]
        ])
        classes = logits2classes(logits, False, threshold=0.1)
        assert classes == [1, 1, 0]

    def test_multiclass_input(self):
        logits = np.array([
            [1, 10, -1],
            [10, -1, 11],
            [-1, -10, -0.1]
        ])
        classes = logits2classes(logits, False)
        assert classes == [1, 2, 2]
