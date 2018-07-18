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

from model import wavenet_estimator
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
      kernel_size: An integer specifying the length of the 1D convolution window.
      data_format: A string, one of `channels_last` or `channels_first`.
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
        channel_axis = 2
    elif data_format == 'channels_first':
        channel_axis = 1
    filters = shape[channel_axis]

    branch = causal_conv1d(
        inputs=inputs,
        filters=2*filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        data_format=data_format
    )

    with tf.variable_scope('filter_and_gate'):
        f, g = tf.split(branch, 2, axis=channel_axis)
        f, g = tf.tanh(f, name='filter'), tf.sigmoid(g, name='gate')

        branch = tf.multiply(f, g, name='activation')

    return tf.layers.conv1d(
        inputs=branch, filters=filters, kernel_size=1,
        data_format=data_format, name='1x1'
    )

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
      data_format: A string, one of `channels_last` or `channels_first`.
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

class WaveNetModel(object):
    """Base class for building a WaveNet model."""

    @staticmethod
    def calculate_receptive_field(kernel_size, dilations):
        """Calculates the receptive field of an output.

        Arguments:
          kernel_size: An integer specifying the length of the 1D convolution windows.
          dilations: List of integers specifying the dilation factor for each of the layers.
            The length of this list also determines the number of layers.

        Returns:
          The receptive field of an output. The number of data points it can theoretically view
          backwards from the input, indirectly through all the hidden layers.
        """
        return (kernel_size-1) * sum(dilations)

    def __init__(self, filters, kernel_size, dilations, output_channels, data_format=None):
        """Creates a WaveNet model.

        Arguments:
          filters: Integer number of filters to use for the residual and skip-connection blocks.
          kernel_size: An integer specifying the length of the 1D convolution windows.
          dilations: List of integers specifying the dilation factor for each of the layers.
            The length of this list also determines the number of layers.
          output_channels: Integer number of outputs.
          data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, length, channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, length)`.
        """
        self.data_format = data_format
        if self.data_format is None:
            self.data_format = 'channels_last'

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.output_channels = output_channels
        self.receptive_field = self.calculate_receptive_field(kernel_size, dilations)

    def __call__(self, inputs, is_training=False):
        """Adds operations to the current graph for the logit output.

        Arguments:
          inputs: A tensor of size
            [batch, length, channels] if data format was set to `channels_last` or
            [batch, channels, length] if data format was set to `channels_first`.
          training: Boolean to indicate if the model is meant to be used for training.
        Returns:
          A `logits` tensor representing unscaled log-probabilities of same size as
          the inputs tensor except the number of channels is modified to output_channels.
        """
        net = inputs

        # Currently unused
        _ = is_training

        with tf.variable_scope('initial'):
            net = causal_conv1d(
                inputs=net, filters=self.filters,
                kernel_size=self.kernel_size, dilation_rate=1,
                data_format=self.data_format
            )

        skip_connections = []
        for i, dilation_rate in enumerate(self.dilations):
            with tf.variable_scope('block_{}_{}'.format(i, dilation_rate)):
                residual = building_block(
                    inputs=net, kernel_size=self.kernel_size,
                    dilation_rate=dilation_rate,
                    data_format=self.data_format
                )

                net += residual
                skip_connections.append(residual)

        with tf.variable_scope('output'):
            net = output_block(skip_connections, self.output_channels, data_format=self.data_format)

        return net

    def model_fn(self, features, labels, mode, params):
        """Works like a model_fn for use with the tf.estimator API.

        The params argument must contain
        Arguments:
          learning_rate: Float learning rate to use with the optimizer.

        Everything else is filled in based on the contents of this instance.
        See wavenet_estimator.model_fn for more information."""
        params.update(self.__dict__)
        params['model'] = self

        return wavenet_estimator.model_fn(features, labels, mode, params)
