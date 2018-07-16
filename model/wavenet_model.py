# Copyright 2018 The Yawn Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for WaveNet as defined in:
[1] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals,
    Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
    WaveNet: A Generative Model for Raw Audio. arXiv:1609.03499
    url: https://arxiv.org/abs/1609.03499
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.ops import causal_conv1d

import tensorflow as tf

def building_block(inputs, kernel_size, data_format, dilation_rate):
    """A single WaveNet block.

    Uses filter and gate activation, output is the last 1x1 convolution
    to be used as a residual and skip connection as defined in section 2.4 of:
      WaveNet: A Generative Model for Raw Audio.
      https://arxiv.org/pdf/1609.03499.pdf
      by Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals,
      Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu

    The number of filters to use is deduced from the input.

    Arguments:
      inputs: Tensor input.
      kernel_size: An integer or tuple/list of a single integer, specifying the
        length of the 1D convolution window.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, length, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, length)`.
      dilation_rate: An integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
    Returns:
      Skip connection output also to be used as a residual.
    """
    shape = inputs.shape.as_list()

    if data_format == 'channels_last':
        filters = shape[2]
    elif data_format == 'channels_first':
        filters = shape[1]

    branch = causal_conv1d(
        inputs=inputs,
        filters=2*filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate
    )

    with tf.variable_scope('filter_and_gate'):
        f, g = tf.split(branch, 2, axis=-1)
        f, g = tf.tanh(f), tf.sigmoid(g)

        branch = tf.multiply(f, g)

    return tf.layers.conv1d(inputs=branch, filters=filters, kernel_size=1, name='1x1')

def output_block(skip_connections, output_channels, data_format, activation=tf.nn.relu):
    """The WaveNet output block.

    Adds up and convolves over the skip connections to produce
    output predictions as defined in section 2.4 of:
      WaveNet: A Generative Model for Raw Audio.
      https://arxiv.org/pdf/1609.03499.pdf
      by Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals,
      Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu

    The output comes in the form of unscaled log-probabilities.

    Arguments:
      skip_connections: List of tensors corresponding to skip contributions.
      output_channels: Integer or tuple/list of single integer, specifying the
        number of channels in the output prediction tensor.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, length, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, length)`.
      activation: Which activation function use, defaults to tf.nn.relu.
    Returns:
      Unscaled log probabilities.
    """
    outputs = tf.add_n(skip_connections)
    outputs = activation(outputs)

    shape = outputs.shape.as_list()

    if data_format == 'channels_last':
        filters = shape[2]
    elif data_format == 'channels_first':
        filters = shape[1]

    outputs = tf.layers.conv1d(
        inputs=outputs, filters=filters, kernel_size=1,
        data_format=data_format, activation=activation,
        name='1x1'
    )

    outputs = tf.layers.conv1d(
        inputs=outputs, filters=output_channels, kernel_size=1,
        data_format=data_format, activation=None,
        name='logits'
    )

    return outputs
