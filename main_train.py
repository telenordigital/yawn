#! /usr/bin/env python
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
"""Trains and exports a saved-model based on some test data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from data.quantized_sine_wave import get_numpy_data, dequantize
from model.wavenet_model import WaveNetModel

# TODO: Figure out some util file for this function.
def data_format_to_shape(
        batch_length=None,
        sequence_length=None,
        channel_length=None,
        data_format='channels_first'
):
    """."""
    shape = [batch_length, None, None]

    channel_axis = 1 if data_format == 'channels_first' else 2
    sequence_axis = 2 if data_format == 'channels_first' else 1

    shape[sequence_axis] = sequence_length
    shape[channel_axis] = channel_length

    return tuple(shape)

def main(FLAGS):
    """."""
    input_channels = 1
    label_channels = 1
    quantization = 64
    num_mixtures = 5

    filters = 8
    kernel_size = 2

    dilation_powers = [0, 1, 2, 3, 4, 5, 6, 7]
    dilations = [kernel_size**power for power in dilation_powers]

    dataset_size = 100000
    data, data_labels, bins = get_numpy_data(dataset_size, quantization, scale=256.0)

    batch_size = 4
    sequence_length = 1024

    dataset_size = (dataset_size//sequence_length)*sequence_length
    data = data[:dataset_size]
    data_labels = data_labels[:dataset_size]

    model = WaveNetModel(
        filters=filters,
        kernel_size=kernel_size,
        dilations=dilations,
        quantization=quantization,
        num_mixtures=num_mixtures,
        bins=bins,
        data_format='channels_last',
        version='mixture'
    )

    data = data.reshape(
        data_format_to_shape(-1, sequence_length, input_channels, data_format=model.data_format)
    )
    data_labels = data_labels.reshape(-1, sequence_length, label_channels)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    classifier = tf.estimator.Estimator(
        model_dir=FLAGS.model_dir,
        model_fn=model.model_fn,
        params=dict(
            learning_rate=1e-4
        ),
        config=tf.estimator.RunConfig(session_config=config)
    )

    classifier.train(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            data, data_labels, batch_size=batch_size, shuffle=True,
            num_epochs=200
        )
    )

    def serving_input_receiver_fn():
        features = tf.placeholder(
            dtype=tf.float32,
            shape=data_format_to_shape(
                None, 1+model.receptive_field, input_channels, data_format=model.data_format
            ),
            name='inputs'
        )
        return tf.estimator.export.TensorServingInputReceiver(
            features=features,
            receiver_tensors=features
        )

    classifier.export_savedmodel(
        export_dir_base='/tmp/wavenet',
        serving_input_receiver_fn=serving_input_receiver_fn
    )
    return 0

if __name__ == '__main__':
    import argparse

    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', type=str, metavar=dir, default=None,
        help='Estimator model directory.'
    )

    args = parser.parse_args()
    exit(main(args))
