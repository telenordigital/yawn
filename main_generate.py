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
"""Imports a trained model and uses it to generate new data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# TODO: Find some util file to store these?
def extract_tensors(signature_def, graph):
    """."""
    output = dict()

    for key in signature_def:
        value = signature_def[key]

        if isinstance(value, tf.TensorInfo):
            output[key] = graph.get_tensor_by_name(value.name)
        else:
            output[key] = extract_operations(value, graph)

    return output

def extract_tags(signature_def, graph):
    """."""
    output = dict()

    for key in signature_def:
        output[key] = dict()
        output[key]['inputs'] = extract_tensors(signature_def[key].inputs, graph)
        output[key]['outputs'] = extract_tensors(signature_def[key].outputs, graph)

    return output

def update_values(session, placeholder, outputs, values, shape):
    """."""
    predictions = session.run(outputs, feed_dict={placeholder : values.reshape(shape)})
    return predictions

def update_plot(session, placeholder, outputs, lines, values, shape):
    """."""
    predictions = update_values(session, placeholder, outputs, values, shape)

    means = predictions['means'][0,-1]
    inverse_standard_deviations = predictions['inverse_standard_deviations'][0,-1]
    coefficients = predictions['coefficients'][0,-1]

    x = np.linspace(-2.0, 2.0, 1001).reshape(-1, 1)
    y = inverse_standard_deviations/np.sqrt(2.0*np.pi) * np.exp(
        -((x-means)*inverse_standard_deviations)**2/2.0
    )

    y = (y*coefficients).sum(axis=-1)

    value = np.random.normal(means, predictions['standard_deviations'][0,-1])
    value = (value*coefficients).sum()

    values[:-1] = values[1:]
    values[-1] = value
    lines[0].set_ydata(values)

    lines[1].set_xdata(x)
    lines[1].set_ydata(y)

    plt.draw()
    plt.pause(1.0/60)

    return values

def main(FLAGS):
    """."""

    with tf.Session(graph=tf.Graph()) as session:
        serve = tf.saved_model.loader.load(session, ['serve'], FLAGS.import_dir)
        tags = extract_tags(serve.signature_def, session.graph)

        placeholder = tags['predictions']['inputs']['input']
        outputs = tags['predictions']['outputs']

        shape = [-1 if s is None else s for s in placeholder.shape.as_list()]
        values = 2.0*np.random.rand(shape[1])-1.0

        plt.ion()
        fig, axes = plt.subplots(2)
        ax1, ax2 = axes

        l1, = ax1.plot(values)
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True)

        l2, = ax2.plot([])
        ax2.set_ylim(-0.1, 4.1)
        ax2.set_xlim(-2.1, 2.1)
        lines = (l1, l2)

        while True:
            try:
                values = update_plot(session, placeholder, outputs, lines, values, shape)
            finally:
                plt.ioff()

    return 0

if __name__ == '__main__':
    import argparse

    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--import_dir', type=str, metavar=dir, required=True,
        help='Where the SavedModel is located.'
    )

    args = parser.parse_args()
    exit(main(args))
