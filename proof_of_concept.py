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

from model.ops import causal_conv1d
from model.wavenet_model import building_block, output_block, WaveNetModel

def build_placeholders(input_channels):
    """."""
    x = tf.placeholder(dtype=tf.float32, shape=(None, input_channels, None))
    y = tf.placeholder(dtype=tf.int32, shape=(None, None, 1))

    return x,y

def build_loss(labels, logits):
    """."""
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def build_train_op(loss, learning_rate=1e-3):
    """."""
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)

def build_model(input_channels, output_channels, filters, kernel_size, dilations):
    """."""
    with tf.variable_scope('inputs'):
        inputs, labels = build_placeholders(input_channels)

    with tf.variable_scope('inference'):
        model = WaveNetModel(
            filters=filters, kernel_size=kernel_size,
            dilations=dilations, output_channels=output_channels,
            data_format='channels_first'
        )
        logits = model(inputs)
        logits = tf.transpose(logits, [0, 2, 1])

    with tf.variable_scope('loss'):
        receptive_field = 0
        for dilation in dilations:
            receptive_field += (kernel_size-1)*dilation
        print('RECEPTIVE FIELD : {}'.format(receptive_field))

        loss = build_loss(labels[:,receptive_field:,...], logits[:,receptive_field:,...])

    with tf.variable_scope('train'):
        train_op = build_train_op(loss)

    with tf.variable_scope('init'):
        init = tf.global_variables_initializer()

    return {
        'x' : inputs,
        'y' : labels,
        'logits' : logits,
        'loss' : loss,
        'train_op' : train_op,
        'init' : init
    }

def report_parameters(graph):
    """."""
    tvars = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    total_parameters = 0
    for tvar in tvars:
        parameters = np.prod(tvar.shape)

        print('{tvar.name} : {tvar.shape}, {parameters}'.format(tvar=tvar, parameters=parameters))
        total_parameters += parameters

    print('=== Total parameters : {} ==='.format(total_parameters))

def quantize(data, bins, right=True):
    """."""
    return np.digitize(data, bins, right=right)

def dequantize(data, bins):
    """."""
    return bins[data]

def train_iteration(session, m, data, labels, batch_num, batch_size):
    """."""
    dataset_size = len(data)
    indices = np.random.randint(dataset_size-batch_size, size=batch_num)

    train_data = []
    train_labels = []
    for index in indices:
        train_data.append(data[index:index+batch_size].reshape(1, -1))
        train_labels.append(labels[index:index+batch_size].reshape(-1, 1))

    loss, _ = session.run(
        [m['loss'], m['train_op']],
        feed_dict={
            m['x'] : train_data,
            m['y'] : train_labels
        }
    )

    return loss

def generate(session, m, values):
    """."""
    predictions = session.run(m['logits'], feed_dict={m['x'] : values.reshape(1, 1, -1)})
    predictions = np.argmax(predictions[0], axis=-1)

    return predictions[-1]

def main():
    """."""
    input_channels = 1
    output_channels = 64

    filters = 16
    kernel_size = 2

    dilation_powers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dilations = [kernel_size**power for power in dilation_powers]

    dataset_size = 100000
    data = np.sin(np.linspace(-256*np.pi, 256*np.pi, dataset_size+1))
    bins = np.percentile(data, np.linspace(0, 100, output_channels))

    labels = quantize(data, bins)
    data = dequantize(labels, bins)

    data = data[:-1]
    labels = labels[1:]

    batch_num = 8
    batch_size = 2048

    graph = tf.Graph()
    with graph.as_default():
        m = build_model(input_channels, output_channels, filters, kernel_size, dilations)
    report_parameters(graph=graph)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as session:
        graph.finalize()

        session.run(m['init'])

        plt.ion()
        fig, ax = plt.subplots()

        values = data[:1024].copy()

        li, = ax.plot(values)
        line = None

        ax.set_ylim(-1.1, 1.1)
        ax.grid(True)

        while True:
            loss = train_iteration(session, m, data, labels, batch_num, batch_size)
            print('loss={}'.format(loss))

            if line is None:
                if loss < 0.1:
                    line = li
                else:
                    continue

            prediction = generate(session, m, values)
            values[:-1] = values[1:]
            values[-1] = dequantize(prediction, bins)

            line.set_ydata(values)
            plt.draw()
            plt.pause(1.0/60)

    return 0

if __name__ == '__main__':
    exit(main())
