from typing import List, Iterable, Tuple, Dict, Union, Optional
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ..utils.parser import parse_tagged_text


# (精准率, 召回率, F值, 数量)
FscoreTuple = Tuple[float, float, float, Optional[int]]


def precision_recall_f_score(
    tp: float, fp: float, fn: float, beta: float = 1
) -> Tuple[float, float, float]:
    """
    根据给定的fp, fp, fn和beta计算精准率, 召回率和F值.

    Parameters
    ----------
    tp: true postive
    fp: false positive
    fn: false negative
    beta: 加权值, f = (1 + beta**2) * (P * R) / (beta**2 * P + R)

    Returns
    ----------
    返回一个tuple, (precision, recall, f score).
    """
    assert beta > 0
    beta2 = beta ** 2
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    return (
        tp / tp_plus_fp if tp_plus_fp else 0,
        tp / tp_plus_fn if tp_plus_fn else 0,
        (1 + beta2) * tp / (tp_plus_fp + beta2 * tp_plus_fn)
    )


def ner_f_score(
    x: Iterable[str], y: Iterable[str], p: Iterable[str], beta: int = 1
) -> Dict[str, FscoreTuple]:
    """
    计算NER结果的F值.

    Parameters
    ----------
    x: 文本
    y: 标注标签
    p: 预测标签
    beta: 加权值, 默认为1, f = (1 + beta**2) * (P * R) / (beta**2 * P + R)

    Returns
    ----------
    返回一个dict.

    key是一类实体或者avg,
    value是一个Tuple, 是对应key的(precision, recall, f score, num samples).
    """
    tp_fp_fn: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for text, label, prediction in zip(x, y, p):
        _text = tuple(text)
        true_entities = parse_tagged_text(_text, label)
        pred_entities = parse_tagged_text(_text, prediction)
        keys = set(true_entities.keys()) | set(pred_entities.keys())
        for key in keys:
            true_set = set(true_entities[key])
            pred_set = set(pred_entities[key])
            tp_fp_fn[key]['tp'] += len(true_set & pred_set)
            tp_fp_fn[key]['fp'] += len(pred_set - true_set)
            tp_fp_fn[key]['fn'] += len(true_set - pred_set)
            tp_fp_fn[key]['num'] += len(true_set)

    scores: Dict[str, FscoreTuple] = dict()
    all_tp, all_fp, all_fn = 0, 0, 0
    for key in tp_fp_fn:
        tp = tp_fp_fn[key]['tp']
        fp = tp_fp_fn[key]['fp']
        fn = tp_fp_fn[key]['fn']
        num = tp_fp_fn[key]['num']
        all_tp += tp
        all_fp += fp
        all_fn += fn
        _p, _r, _f = precision_recall_f_score(tp, fp, fn)
        scores[key] = (_p, _r, _f, num)
    if all_tp == all_tp == all_fn == 0:
        scores['avg'] = (0.0, 0.0, 0.0, None)
    else:
        _p, _r, _f = precision_recall_f_score(all_tp, all_fp, all_fn)
        scores['avg'] = (_p, _r, _f, None)
    return scores


def classify_f_score(
    y: List[List[int]], p: List[List[int]], is_multilabel: bool,
    beta: float = 1, labels: Optional[List[str]] = None
) -> Dict[Union[str, int], FscoreTuple]:
    """
    计算分类结果的F值.

    Parameters
    ----------
    y: 标注标签
    p: 预测标签
    is_multilabel: 是否是多标签分类
    beta: 加权值, 默认为1, f = (1 + beta**2) * (P * R) / (beta**2 * P + R)
    labels: 标签名, 如果不提供则取0...n_classes - 1

    Returns
    ----------
    返回一个dict.

    如果是非多标签的二分类问题, 则仅包含score一个key, value是一个Tuple
    是正例的(precision, recall, f score, num samples).

    其他情况下, key是一个类别或者avg, value是一个Tuple,
    是对应key的(precision, recall, f score, num samples).
    """
    scores: Dict[Union[str, int], FscoreTuple] = dict()
    _y, _p = np.array(y), np.array(p)
    assert _y.shape == _p.shape, 'y and p must have same shape'

    max_class_idx = _y.max()
    if max_class_idx <= 1 and not is_multilabel:
        score = precision_recall_fscore_support(_y, _p, average='binary')
        scores['score'] = score
    else:
        num_classes = _y.shape[1] if _y.ndim == 2 else max_class_idx + 1
        idx = list(range(num_classes))
        detail_score = precision_recall_fscore_support(_y, _p, labels=idx)
        micro_score = precision_recall_fscore_support(
            _y, _p, labels=idx, average='micro'
        )
        for i, score in enumerate(zip(*detail_score)):
            if labels is None:
                scores[i] = score
            else:
                scores[labels[i]] = score
        scores['avg'] = micro_score
    return scores
