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

from data.sine_wave import get_numpy_data, dequantize
from model.wavenet_model import WaveNetModel

def main():
    """."""
    input_channels = 1
    output_channels = 64

    filters = 16
    kernel_size = 2

    dilation_powers = [0, 1, 2, 3, 4, 5, 6, 7]
    dilations = [kernel_size**power for power in dilation_powers]

    dataset_size = 100000
    data, data_labels, bins = get_numpy_data(dataset_size, output_channels, scale=256)

    batch_size = 4
    sequence_length = 1024

    dataset_size = (dataset_size//sequence_length)*sequence_length
    data = data[:dataset_size]
    data_labels = data_labels[:dataset_size]

    data = data.reshape(-1, sequence_length, 1)
    data_labels = data_labels.reshape(-1, sequence_length, 1)

    model = WaveNetModel(
        filters=filters,
        kernel_size=kernel_size,
        dilations=dilations,
        output_channels=output_channels,
        data_format='channels_last'
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    classifier = tf.estimator.Estimator(
        model_fn=model.model_fn,
        params=dict(
            learning_rate=1e-3
        ),
        config=tf.estimator.RunConfig(session_config=config)
    )

    classifier.train(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            data, data_labels, batch_size=batch_size, shuffle=True,
            num_epochs=100
        )
    )

    def serving_input_receiver_fn():
        features = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        return tf.estimator.export.TensorServingInputReceiver(
            features=features,
            receiver_tensors=features
        )

    classifier.export_savedmodel('/tmp/wavenet', serving_input_receiver_fn)

    return 0

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    exit(main())
