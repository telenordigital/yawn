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
"""."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.sine_wave import get_numpy_data, dequantize
from model.wavenet_model import WaveNetModel

def build_placeholders(input_channels, label_channels=1, sequence_length=None, data_format='channels_last'):
    """."""
    channel_axis = 1 if data_format == 'channels_first' else 2
    sequence_axis = 2 if data_format == 'channels_first' else 1

    input_shape = [None, None, None]
    input_shape[channel_axis] = input_channels
    input_shape[sequence_axis] = sequence_length

    features = tf.placeholder(dtype=tf.float32, shape=input_shape, name='features')
    labels = tf.placeholder(dtype=tf.int32, shape=(None, sequence_length, label_channels), name='labels')

    return features, labels

def report_parameters(graph):
    """."""
    tvars = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    total_parameters = 0
    for tvar in tvars:
        parameters = np.prod(tvar.shape)

        print('{tvar.name} : {tvar.shape}, {parameters}'.format(tvar=tvar, parameters=parameters))
        total_parameters += parameters

    print('=== Total parameters : {} ==='.format(total_parameters))

def train_iteration(session, model, features, labels, data, data_labels, batch_size, sequence_length):
    """."""
    dataset_size = len(data)
    indices = np.random.randint(dataset_size-sequence_length, size=batch_size)

    train_data = []
    train_labels = []
    for index in indices:
        train_data.append(data[index:index+sequence_length].reshape(-1, 1))
        train_labels.append(data_labels[index:index+sequence_length].reshape(-1, 1))

    loss, _ = session.run(
        [model.loss, model.train_op],
        feed_dict={
            features : train_data,
            labels : train_labels
        }
    )

    return loss

def generate(session, features, model, values):
    """."""
    predictions = session.run(
        model.predictions['predictions'],
        feed_dict={features : values.reshape(1, -1, 1)}
    )

    # Return only the last prediction
    return predictions[0,-1]

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

    graph = tf.Graph()
    with graph.as_default():
        features, labels = build_placeholders(input_channels=input_channels)

        model = WaveNetModel(
            filters=filters,
            kernel_size=kernel_size,
            dilations=dilations,
            output_channels=output_channels,
            data_format='channels_last'
        )

        receptive_field = model.receptive_field
        print('Receptive field : {}'.format(receptive_field))

        with tf.variable_scope('wavenet', reuse=False):
            train_model = model.model_fn(
                features, labels, tf.estimator.ModeKeys.TRAIN, dict(learning_rate=1e-3)
            )

        with tf.variable_scope('wavenet', reuse=True):
            infer_model = model.model_fn(
                features, None, tf.estimator.ModeKeys.PREDICT, dict()
            )

        init_op = tf.global_variables_initializer()
    graph.finalize()

    report_parameters(graph=graph)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as session:
        session.run(init_op)

        plt.ion()
        fig, ax = plt.subplots()

        values = data[:1+receptive_field].copy()

        li, = ax.plot(values)
        line = None

        ax.set_ylim(-1.1, 1.1)
        ax.grid(True)

        while True:
            loss = train_iteration(
                session, train_model, features, labels,
                data, data_labels, batch_size, sequence_length
            )
            print('loss={}'.format(loss))

            if line is None:
                if loss < 1.0:
                    line = li
                else:
                    continue

            prediction = generate(session, features, infer_model, values)
            values[:-1] = values[1:]
            values[-1] = dequantize(prediction, bins)

            line.set_ydata(values)
            plt.draw()
            plt.pause(1.0/60)

    return 0

if __name__ == '__main__':
    exit(main())
