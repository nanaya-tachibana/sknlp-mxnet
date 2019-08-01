import numpy as np
from sknlp.metric.f_score import precision_recall_f_score
from sknlp.metric import ner_f_score, classify_f_score


class TestPrecisionRecallFscore:

    def test_normal_input(self):
        tp, fp, fn = 4, 2, 1
        np.testing.assert_allclose(
            precision_recall_f_score(tp, fp, fn),
            (4 / 6, 4 / 5, 8 / 11)
        )

    def test_beta_equal_two(self):
        tp, fp, fn = 4, 2, 1
        np.testing.assert_allclose(
            precision_recall_f_score(tp, fp, fn, beta=2),
            (4 / 6, 4 / 5, 10 / 13)
        )

    def test_recall_equal_zero(self):
        tp, fp, fn = 0, 2, 0
        np.testing.assert_allclose(
            precision_recall_f_score(tp, fp, fn, beta=2),
            (0, 0, 0)
        )


class TestNERFscore:

    def test_normal_input(self):
        x = ['哦上海你北京哦邓大平啦啦啦', '苏州市猫泽东']
        y = [['O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC',
              'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O'],
             ['B-LOC', 'I-LOC', 'I-LOC', 'B-PER', 'I-PER', 'I-PER']]
        p = [['O', 'O', 'O', 'O', 'B-LOC', 'I-LOC',
              'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O'],
             ['B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER']]
        score = ner_f_score(x, y, p)
        np.testing.assert_allclose(score['LOC'], (1, 2 / 3, 4 / 5, 3))
        np.testing.assert_allclose(score['PER'], (1 / 2, 1 / 2, 1 / 2, 2))
        np.testing.assert_allclose(score['avg'][:-1], (3 / 4, 3 / 5, 2 / 3))
        assert score['avg'][-1] is None

    def test_all_O_input(self):
        x = ['哦啦啦啦', '啊啊啊']
        y = [['O', 'O', 'O', 'O'],
             ['O', 'O', 'O']]
        p = y.copy()
        score = ner_f_score(x, y, p)
        assert 'LOC' not in score
        np.testing.assert_allclose(score['avg'][:-1], (0, 0, 0))
        assert score['avg'][-1] is None

    def test_tuple_input(self):
        x = [(0, 10, 11, 2, 12, 13, 0, 14, 15, 16, 1, 1, 1),
             (17, 18, 19, 20, 21, 22)]
        y = [['O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC',
              'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O'],
             ['B-LOC', 'I-LOC', 'I-LOC', 'B-PER', 'I-PER', 'I-PER']]
        p = [['O', 'O', 'O', 'O', 'B-LOC', 'I-LOC',
              'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O'],
             ['B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER']]
        score = ner_f_score(x, y, p)
        np.testing.assert_allclose(score['LOC'], (1, 2 / 3, 4 / 5, 3))
        np.testing.assert_allclose(score['PER'], (1 / 2, 1 / 2, 1 / 2, 2))
        np.testing.assert_allclose(score['avg'][:-1], (3 / 4, 3 / 5, 2 / 3))
        assert score['avg'][-1] is None


class TestClassifyFscore:

    def test_multiclass_multilabel_input(self):
        y = [[0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        p = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
        labels = ['cat', 'dog', 'horse']
        score = classify_f_score(y, p, True, labels=labels)
        np.testing.assert_allclose(score['cat'], (1 / 2, 1 / 2, 1 / 2, 2))
        np.testing.assert_allclose(score['dog'], (1 / 2, 1 / 3, 2 / 5, 3))
        np.testing.assert_allclose(score['horse'], (1 / 2, 1, 2 / 3, 1))
        np.testing.assert_allclose(score['avg'][:-1], (1 / 2, 1 / 2, 1 / 2))
        assert score['avg'][-1] is None

    def test_input_without_labels(self):
        y = [[0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        p = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
        score = classify_f_score(y, p, True)
        np.testing.assert_allclose(score[0], (1 / 2, 1 / 2, 1 / 2, 2))
        np.testing.assert_allclose(score[1], (1 / 2, 1 / 3, 2 / 5, 3))
        np.testing.assert_allclose(score[2], (1 / 2, 1, 2 / 3, 1))
        np.testing.assert_allclose(score['avg'][:-1], (1 / 2, 1 / 2, 1 / 2))
        assert score['avg'][-1] is None

    def test_multiclass_input(self):
        y = [2, 0, 1, 1]
        p = [2, 1, 0, 1]
        score = classify_f_score(y, p, False)
        np.testing.assert_allclose(score[0], (0, 0, 0, 1))
        np.testing.assert_allclose(score[1], (1 / 2, 1 / 2, 1 / 2, 2))
        np.testing.assert_allclose(score[2], (1, 1, 1, 1))
        np.testing.assert_allclose(score['avg'][:-1], (1 / 2, 1 / 2, 1 / 2))
        assert score['avg'][-1] is None

    def test_binary_input(self):
        y = [1, 0, 0, 1]
        p = [1, 1, 0, 1]
        score = classify_f_score(y, p, False)
        np.testing.assert_allclose(score['score'][:-1], (2 / 3, 1, 4 / 5))
        assert score['score'][-1] is None
