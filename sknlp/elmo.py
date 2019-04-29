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

from typing import Tuple
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class AdaptiveSoftmax(gluon.HybridBlock):

    def __init__(self, input_size: int, num_classes: int,
                 cutoffs: Tuple[int], div_factor: int = 4,
                 weight_initializer: mx.init.Initializer = None, **kwargs):
        super().__init__(**kwargs)

        self._num_clusters = len(cutoffs)
        projection_sizes = [max(1, input_size // div_factor ** i)
                            for i in range(self._num_clusters)]
        with self.name_scope():
            head_size = cutoffs[0] + self._num_clusters
            self.head_layer = nn.Dense(
                head_size, flatten=False, use_bias=False,
                weight_initializer=weight_initializer,
                prefix='head_')
            self.tail_layers = []
            for i in range(self._num_clusters):
                tail_layer = nn.Sequential(prefix=f'tail{i}_')
                with tail_layer.name_scope():
                    tail_layers = nn.Dense(


    def hybrid_forward(self):
        pass
