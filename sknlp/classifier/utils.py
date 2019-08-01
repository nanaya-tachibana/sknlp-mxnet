from typing import Union, List
import numpy as np
from scipy.special import expit


def logits2classes(
    logits: np.ndarray, is_multilabel: bool,
    threshold: Union[np.ndarray, float] = 0.5
) -> Union[List[int], List[List[int]]]:
    """
    根据给定的`threshold`, 将分类问题的`logits`解析为对应的类别.

    Parameters
    ----------
    logits: 输入的`logits`, shape(`n_samples`, `n_classes`)
    is_multilabel: 是否为多标签分类
    threshold: 正例判断的阈值(仅在多标签和二分类问题时有作用)

    Returns
    ----------
    返回一个长度为`n_samples`的``list``.

    如果是多标签分类, 每一个sample对应的结果为一个``list``,
    其中的每个``int``值为这个sample对应的类别.

    如果不是多标签分类, 则每一个sample对应的结果为一个``int``,
    表示这个sample对应的类别.
    """
    assert logits.ndim == 2
    num_classes = logits.shape[1]
    if is_multilabel:
        return [
            np.where(expit(logits[i, :]) > threshold)[0].tolist()
            for i in range(logits.shape[0])
        ]
    elif num_classes == 2:
        return np.where(expit(logits[:, 1]) > threshold, 1, 0).tolist()
    else:
        return np.argmax(logits, axis=1).tolist()
