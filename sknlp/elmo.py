# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Bidirectional LM encoder."""

from typing import List, Union, Optional
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


HybridType = Union[mx.ndarray.NDArray, mx.symbol.Symbol]


class AdaptiveSoftmax(gluon.HybridBlock):

    def __init__(self, input_size: int, num_classes: int,
                 cutoffs: List[int], div_factor: int = 4,
                 weight_initializer: Optional[mx.init.Initializer] = None,
                 **kwargs) -> None:
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
        projection_sizes = [max(1, input_size // div_factor ** (i + 1))
                            for i in range(self._num_clusters)]

        if isinstance(cutoffs, list):
            self._cutoffs = cutoffs
        else:
            self._cutoffs = list(cutoffs)
        self._cutoffs.append(num_classes)

        with self.name_scope():
            head_size = self._cutoffs[0] + self._num_clusters
            self.head_layer = nn.Dense(
                head_size, flatten=False, use_bias=False,
                weight_initializer=weight_initializer,
                prefix='head_')
            for i in range(self._num_clusters):
                self.tail_layers: List[mx.gluon.HybridBlock] = []
                tail_layer = nn.HybridSequential(prefix=f'tail{i}_')
                with tail_layer.name_scope():
                    tail_layer.add(nn.Dense(
                        projection_sizes[i], flatten=False, use_bias=False,
                        weight_initializer=weight_initializer,
                        prefix='proj_'))
                    tail_layer.add(nn.Dense(
                        self._cutoffs[i + 1] - self._cutoffs[i],
                        flatten=False, use_bias=False,
                        weight_initializer=weight_initializer, prefix='w_'))
                self.tail_layers.append(tail_layer)
                # Note that Blocks inside the list, tuple or dict will
                # not be registered automatically
                self.register_child(tail_layer)
            self.softmaxce = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    def hybrid_forward(self, F, inputs: HybridType,
                       labels: HybridType) -> HybridType:
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
            mask = F.logical_and(F.greater_equal(labels, self._cutoffs[i]),
                                 F.lesser(labels, self._cutoffs[i + 1]))
            # update head labels
            head_labels = F.where(mask, ones * (self._cutoffs[0] + i),
                                  head_labels)
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
