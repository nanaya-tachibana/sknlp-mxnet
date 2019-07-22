import numpy as np
from sknlp.metric.f_score import precision_recall_f_score, ner_f_score


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
        np.testing.assert_allclose(score['LOC'], (1, 2 / 3, 4 / 5))
        np.testing.assert_allclose(score['PER'], (1 / 2, 1 / 2, 1 / 2))
        np.testing.assert_allclose(score['avg'], (3 / 4, 3 / 5, 2 / 3))

    def test_all_O_input(self):
        x = ['哦啦啦啦', '啊啊啊']
        y = [['O', 'O', 'O', 'O'],
             ['O', 'O', 'O']]
        p = y.copy()
        score = ner_f_score(x, y, p)
        assert 'LOC' not in score
        np.testing.assert_allclose(score['avg'], (0, 0, 0))

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
        np.testing.assert_allclose(score['LOC'], (1, 2 / 3, 4 / 5))
        np.testing.assert_allclose(score['PER'], (1 / 2, 1 / 2, 1 / 2))
        np.testing.assert_allclose(score['avg'], (3 / 4, 3 / 5, 2 / 3))
