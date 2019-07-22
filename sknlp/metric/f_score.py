from collections import defaultdict

from ..utils.parser import parse_tagged_text


def precision_recall_f_score(tp, fp, fn, beta=1):
    assert beta > 0
    beta2 = beta ** 2
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    return (
        tp / tp_plus_fp if tp_plus_fp else 0,
        tp / tp_plus_fn if tp_plus_fn else 0,
        (1 + beta2) * tp / (tp_plus_fp + beta2 * tp_plus_fn)
    )


def ner_f_score(x, y, p, beta=1):
    tp_fp_fn = defaultdict(lambda: defaultdict(int))
    for text, label, prediction in zip(x, y, p):
        text = tuple(text)
        true_entities = parse_tagged_text(text, label)
        pre_entities = parse_tagged_text(text, prediction)
        keys = set(true_entities.keys()) | set(pre_entities.keys())
        for key in keys:
            t = set(true_entities[key])
            p = set(pre_entities[key])
            tp_fp_fn[key]['tp'] += len(t & p)
            tp_fp_fn[key]['fp'] += len(p - t)
            tp_fp_fn[key]['fn'] += len(t - p)

    scores = dict()
    all_tp, all_fp, all_fn = 0, 0, 0
    for key in tp_fp_fn:
        tp = tp_fp_fn[key]['tp']
        fp = tp_fp_fn[key]['fp']
        fn = tp_fp_fn[key]['fn']
        all_tp += tp
        all_fp += fp
        all_fn += fn
        scores[key] = precision_recall_f_score(tp, fp, fn)
    if all_tp == all_tp == all_fn == 0:
        scores['avg'] = (0, 0, 0)
    else:
        scores['avg'] = precision_recall_f_score(all_tp, all_fp, all_fn)
    return scores
