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
from model.wavenet_model import building_block, output_block

def build_placeholders(input_channels):
    """."""
    x = tf.placeholder(dtype=tf.float32, shape=(None, None, input_channels))
    y = tf.placeholder(dtype=tf.int32, shape=(None, None, 1))

    return x,y

def build_inference(inputs, filters, kernel_size, dilation_powers, output_channels):
    """."""
    net = inputs

    with tf.variable_scope('input'):
        net = causal_conv1d(inputs=net, filters=filters, kernel_size=kernel_size)

    residuals = []
    for i, dilation_power in enumerate(dilation_powers):
        dilation_rate = kernel_size**dilation_power

        with tf.variable_scope('residual_{}_{}'.format(i, dilation_rate)):
            residual = building_block(
                inputs=net, kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                data_format='channels_last'
            )

            net += residual
            residuals.append(residual)

    with tf.variable_scope('output'):
        net = output_block(residuals, output_channels, data_format='channels_last')

    return net

def build_loss(labels, logits):
    """."""
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def build_train_op(loss, learning_rate=1e-2):
    """."""
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)

def build_model(input_channels, output_channels, filters, kernel_size, dilation_powers):
    """."""
    with tf.variable_scope('inputs'):
        inputs, labels = build_placeholders(input_channels)

    with tf.variable_scope('inference'):
        logits = build_inference(inputs, filters, kernel_size, dilation_powers, output_channels)

    with tf.variable_scope('loss'):
        loss = build_loss(labels, logits)

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
        train_data.append(data[index:index+batch_size].reshape(-1, 1))
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
    predictions = session.run(m['logits'], feed_dict={m['x'] : values.reshape(1, -1, 1)})
    predictions = np.argmax(predictions[0], axis=-1)

    return predictions[-1]

def main():
    """."""
    input_channels = 1
    output_channels = 64

    filters = 16
    kernel_size = 2

    dilation_powers = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    dataset_size = 100000
    data = np.sin(np.linspace(-256*np.pi, 256*np.pi, dataset_size+1))
    bins = np.linspace(-1, 1, output_channels)

    labels = quantize(data, bins)
    data = dequantize(labels, bins)

    data = data[:-1]
    labels = labels[1:]

    batch_num = 8
    batch_size = 1024

    graph = tf.Graph()
    with graph.as_default():
        m = build_model(input_channels, output_channels, filters, kernel_size, dilation_powers)
    graph.finalize()
    report_parameters(graph=graph)

    with tf.Session(graph=graph) as session:
        session.run(m['init'])

        plt.ion()
        fig, ax = plt.subplots()

        values = data[:512].copy()

        li, = ax.plot(values)
        line = None

        ax.set_ylim(-1.1, 1.1)
        ax.grid(True)

        while True:
            loss = train_iteration(session, m, data, labels, batch_num, batch_size)
            print('loss={}'.format(loss))

            if line is None:
                if loss < 1.0:
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
