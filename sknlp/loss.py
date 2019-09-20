from typing import List, Union, Optional
import functools
import math
import mxnet as mx
from mxnet.gluon import nn


HybridType = Union[mx.ndarray.NDArray, mx.symbol.Symbol]


class AdaptiveSoftmax(nn.HybridBlock):
    """
    Parameters
    ----------
    input_size: int
    输入数据维度
    num_classes: int
    类别数
    cutoffs: List[int]
    类别分组, 如[10, 1000]表示将所有类别分为
    [0, 10), [10, 1000)和[1000, num_classes)三个组,
    不同分组的隐层维度不同
    div_factor: int, optional
    数据维度衰减因子, 用于压缩后面的类别分组的隐层维度,
    如对于第二个分组隐层的维度为``input_size`` // ``div_factor``
    weight_initializer: mx.init.Initializer
    隐层权重初始化函数
    """

    def __init__(
        self, input_size: int, num_classes: int, cutoffs: List[int],
        div_factor: int = 4, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        cutoffs = list(cutoffs)
        if (cutoffs != sorted(cutoffs)
            or min(cutoffs) < 0
            or max(cutoffs) > num_classes
            or len(set(cutoffs)) != len(cutoffs)
                or any([int(c) != c for c in cutoffs])):
            raise ValueError('cutoffs should be an ordered list of unique, '
                             'positive integers, where each value is between'
                             ' 1 and num_classes - 1')
        self._num_clusters = len(cutoffs)
        projection_sizes = [
            max(1, input_size // div_factor ** (i + 1))
            for i in range(self._num_clusters)
        ]

        if isinstance(cutoffs, list):
            self._cutoffs = cutoffs
        else:
            self._cutoffs = list(cutoffs)
        self._cutoffs.append(num_classes)

        with self.name_scope():
            head_size = self._cutoffs[0] + self._num_clusters
            self.head_layer = nn.Dense(
                head_size, flatten=False, use_bias=False,
                weight_initializer=mx.init.Normal(1 / math.sqrt(input_size)),
                prefix='head_', in_units=input_size
            )
            self.tail_layers: List[mx.gluon.HybridBlock] = []
            for i in range(self._num_clusters):
                tail_layer = nn.HybridSequential(prefix=f'tail{i}_')
                with tail_layer.name_scope():
                    tail_layer.add(nn.Dense(
                        projection_sizes[i], flatten=False, use_bias=False,
                        prefix='proj_', in_units=input_size
                    ))
                    tail_layer.add(nn.Dense(
                        self._cutoffs[i + 1] - self._cutoffs[i],
                        flatten=False, use_bias=False,
                        prefix='w_', in_units=projection_sizes[i]
                    ))
                self.tail_layers.append(tail_layer)
                # Note that Blocks inside the list, tuple or dict will
                # not be registered automatically
                self.register_child(tail_layer)
            self.softmaxce = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    def hybrid_forward(
        self, F, input: HybridType, labels: HybridType, help_idx: HybridType,
    ) -> HybridType:
        '''
        Parameters
        ----------
        input: shape(batch_size, ``num_classes``)
        labels: shape(batch_size, )
        help_idx: shape(batch_size, )
        '''
        def update_tail_loss(tail_loss, mask, tail_layer, offset):
            # compute tail loss
            # shape(sum(mask), num_classes)
            tail_input = F.contrib.boolean_mask(input, mask)
            # shape(batch_size, cutoffs[i + 1] - cutoffs[i])
            tail_logits = tail_layer(tail_input)
            # shape(sum(mask), )
            tail_labels = F.contrib.boolean_mask(
                labels - offset, mask
            )
            idx = F.contrib.boolean_mask(help_idx, mask)
            pred = -F.log_softmax(tail_logits, axis=-1)
            return F.contrib.index_copy(
                tail_loss, idx, F.pick(pred, tail_labels)
            )

        head_labels = labels
        ones = F.ones_like(labels)
        tail_loss = F.zeros_like(labels)
        for i in range(self._num_clusters):
            mask = F.broadcast_logical_and(
                F.broadcast_greater_equal(labels, ones * self._cutoffs[i]),
                F.broadcast_lesser(labels, ones * self._cutoffs[i + 1])
            )
            # update head labels
            head_labels = F.where(
                mask, ones * (self._cutoffs[0] + i), head_labels
            )
            tail_loss = F.contrib.cond(
                F.sum(mask) > 0,
                functools.partial(
                    update_tail_loss, tail_loss, mask,
                    self.tail_layers[i], self._cutoffs[i]
                ),
                lambda: tail_loss
            )

        head_logits = self.head_layer(input)
        head_loss = self.softmaxce(head_logits, head_labels)
        return head_loss + tail_loss


class ElmoLoss(nn.HybridBlock):

    def __init__(self, loss_func, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.loss_func = loss_func

    def hybrid_forward(
        self, F,
        input: HybridType,
        mask: HybridType,
        forward_labels: HybridType,
        backward_labels: HybridType,
        help_idx: HybridType
    ) -> HybridType:
        # can't work with ndarray now
        # last_layer = F.SequenceLast(input)
        last_layer = F.squeeze(
            F.slice_axis(input, begin=-1, end=None, axis=0), axis=0
        )
        forward_input, backward_input = F.split(
            last_layer, axis=-1, num_outputs=2
        )
        forward_backward_input = F.concat(
            F.reshape(forward_input, shape=(-3, -1)),
            F.reshape(backward_input, shape=(-3, -1)),
            dim=0
        )
        forward_backward_labels = F.concat(
            F.reshape(forward_labels, shape=(-1, )),
            F.reshape(backward_labels, shape=(-1, )),
            dim=0
        )
        mask = F.reshape(F.tile(mask, reps=(2, 1)), shape=(-1, ))
        return 0.5 * self.loss_func(
            forward_backward_input, forward_backward_labels, help_idx
        ) * mask
