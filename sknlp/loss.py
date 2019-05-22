from typing import List, Union, Optional
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


HybridType = Union[mx.ndarray.NDArray, mx.symbol.Symbol]


class FullSoftmax(gluon.HybridBlock):

    def __init__(
        self, num_classes: int,
        weight_initializer: Optional[mx.init.Initializer] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        with self.name_scope():
            self.dense_layer = nn.Dense(
                num_classes, flatten=False,
                weight_initializer=weight_initializer, prefix='proj_'
            )
            self.softmaxce = gluon.loss.SoftmaxCrossEntropyLoss()

    def hybrid_forward(
        self, F, inputs: HybridType, labels: HybridType
    ) -> HybridType:
        logits = self.dense_layer(inputs)
        return self.softmaxce(logits, labels)


class AdaptiveSoftmax(gluon.HybridBlock):
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
        div_factor: int = 4,
        weight_initializer: Optional[mx.init.Initializer] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

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
                weight_initializer=weight_initializer, prefix='head_'
            )
            for i in range(self._num_clusters):
                self.tail_layers: List[mx.gluon.HybridBlock] = []
                tail_layer = nn.HybridSequential(prefix=f'tail{i}_')
                with tail_layer.name_scope():
                    tail_layer.add(nn.Dense(
                        projection_sizes[i], flatten=False, use_bias=False,
                        weight_initializer=weight_initializer, prefix='proj_'
                    ))
                    tail_layer.add(nn.Dense(
                        self._cutoffs[i + 1] - self._cutoffs[i],
                        flatten=False, use_bias=False,
                        weight_initializer=weight_initializer, prefix='w_'
                    ))
                self.tail_layers.append(tail_layer)
                # Note that Blocks inside the list, tuple or dict will
                # not be registered automatically
                self.register_child(tail_layer)
            self.softmaxce = gluon.loss.SoftmaxCrossEntropyLoss()

    def hybrid_forward(
        self, F, inputs: HybridType, labels: HybridType
    ) -> HybridType:
        '''
        Parameters
        ----------
        inputs: shape(batch_size, ``num_classes``)
        labels: shape(batch_size, )
        '''
        head_labels = labels
        ones = F.ones_like(labels)
        tail_loss = F.zeros_like(labels)
        for i in range(self._num_clusters):
            mask = F.logical_and(
                F.greater_equal(labels, self._cutoffs[i]),
                F.lesser(labels, self._cutoffs[i + 1])
            )
            # update head labels
            head_labels = F.where(
                mask, ones * (self._cutoffs[0] + i), head_labels
            )
            # compute tail loss
            # shape(sum(mask), num_classes)
            # tail_inputs = F.contrib.boolean_mask(inputs, mask)
            # shape(batch_size, cutoffs[i + 1] - cutoffs[i])
            tail_logits = self.tail_layers[i](inputs)
            # shape(sum(mask), )
            # tail_labels = F.contrib.boolean_mask(labels - self._cutoffs[i], mask)
            tail_labels = labels - self._cutoffs[i]
            pred = F.log_softmax(tail_logits, axis=-1)
            F.where(mask, -F.pick(pred, tail_labels), tail_loss, out=tail_loss)
        head_logits = self.head_layer(inputs)
        head_loss = self.softmaxce(head_logits, head_labels)
        return head_loss + tail_loss
