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
"""Common operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

SUPPORTED_DATA_FORMATS = (
    'channels_last',
    'channels_first'
)

def causal_conv1d(inputs,
                  filters,
                  kernel_size,
                  data_format='channels_last',
                  dilation_rate=1,
                  padding_value=0,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=None,
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  trainable=True,
                  name=None,
                  reuse=None):
    """1D convolution layer with padding along the data direction.

    This layer operates just like a tf.layers.conv1d layer except that
    the output is shifted such that a causal relationship is maintained
    along the data axis.

    If the `channels_last` data_format is used this means that each `outputs[:,i,:]`
    may depend at most on the previous `i` inputs, `inputs[:,:1+i,:]`.

    The output is padded after the activation function has been applied to the output.

    Arguments:
      inputs: Tensor input.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer, specifying the
        length of the 1D convolution window.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, length, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, length)`.
      dilation_rate: An integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
      padding_value: Value to pad output with, it is added after the activation
        function has been applied.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: A string, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      Causally padded tensor.
    """
    if data_format not in SUPPORTED_DATA_FORMATS:
        raise ValueError('Data format {} not supported'.format(data_format))

    if name is None:
        name = 'causal_conv1d'

    # Figure out the size and layout of the padding
    required_padding = (kernel_size-1)*dilation_rate
    if data_format == 'channels_last':
        paddings = ((0, 0), (required_padding, 0), (0, 0))
    elif data_format == 'channels_first':
        paddings = ((0, 0), (0, 0), (required_padding))

    tensor = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='valid',
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name='{}/conv1d'.format(name),
        reuse=reuse
    )

    return tf.pad(
        tensor=tensor,
        paddings=paddings,
        mode='CONSTANT',
        name='{}/shifted'.format(name),
        constant_values=padding_value
    )
