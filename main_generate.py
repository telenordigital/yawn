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

from data.quantization import requantize, dequantize
from data.existing_numpy_data import get_numpy_data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy import special

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

def probability_setup(n_values, bins):
    """."""
    fig, axes = plt.subplots(2)
    ax1, ax2 = axes

    l1, = ax1.plot(np.zeros(n_values))
    ax1.set_ylim(bins[0]-0.1, bins[-1]+0.1)
    ax1.grid(True)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$v$')

    l2, = ax2.plot([], [], '--bo', markersize=2.0, linewidth=1.0)
    ax2.grid(True)
    ax2.set_xlim(bins[0]-0.1, bins[-1]+0.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel('$v$')
    ax2.set_ylabel('$p$')

    return fig, axes, (l1, l2)

def mixture_update(session, placeholder, outputs, lines, values, shape):
    """."""
    predictions = session.run(outputs, feed_dict={placeholder : values.reshape(shape)})

    # Unpack values since we're only going to use batch 0 and last element -1
    means = predictions['means'][0,-1]
    inverse_standard_deviations = predictions['inverse_standard_deviations'][0,-1]
    standard_deviations = predictions['standard_deviations'][0,-1]
    coefficients = predictions['coefficients'][0,-1]
    bins = predictions['bins']

    # Bottom plot with probabilities
    # x values are the bin mid-points
    # y values are the probabilities of observing the value in that bin
    reshaped_bins = bins.reshape(-1, 1)
    x = 0.5*(bins[1:] + bins[:-1])
    y = 0.5*(1.0 + special.erf((reshaped_bins-means)*inverse_standard_deviations*np.sqrt(0.5)))
    y[0] = 0.0
    y[-1] = 1.0
    y = np.diff(y, axis=0)
    y = (y*coefficients).sum(axis=-1)

    # Top plot with the predicted values
    index = np.random.multinomial(1, coefficients).argmax()
    value = np.random.normal(means[index], standard_deviations[index])

    # Alternate plot is mean of mixture
    # value = means
    # value = (value*coefficients).sum()

    # Cycle new value into buffer at the end
    values[:-1] = values[1:]
    values[-1] = requantize(value, bins)

    lines[0].set_ydata(values)
    lines[1].set_data(x, y)

    plt.draw()
    plt.pause(1.0/60)

    return values

def categorical_update(session, placeholder, outputs, lines, values, shape):
    """."""
    predictions = session.run(outputs, feed_dict={placeholder : values.reshape(shape)})

    probabilities = predictions['probabilities'][0, -1]
    bins = predictions['bins']

    x = 0.5*(bins[1:] + bins[:-1])
    y = probabilities

    value = np.random.choice(bins[:-1], p=probabilities)

    values[:-1] = values[1:]
    values[-1] = requantize(value, bins)

    lines[0].set_ydata(values)
    lines[1].set_data(x, y)

    plt.draw()
    plt.pause(1.0/60)

    return values

VERSIONS = {
    'mixture' : (probability_setup, mixture_update),
    'categorical' : (probability_setup, categorical_update)
}

def main(FLAGS):
    """."""

    with tf.Session(graph=tf.Graph()) as session:
        serve = tf.saved_model.loader.load(session, ['serve'], FLAGS.import_dir)
        tags = extract_tags(serve.signature_def, session.graph)

        placeholder = tags['predictions']['inputs']['input']
        outputs = tags['predictions']['outputs']

        if 'coefficients' in outputs:
            version = 'mixture'
        else:
            version = 'categorical'
        setup, update = VERSIONS[version]

        bins = session.run(outputs['bins'])

        shape = [-1 if s is None else s for s in placeholder.shape.as_list()]
        values = np.random.choice(bins, size=shape[1])

        plt.ion()
        fig, axes, lines = setup(n_values=shape[1], bins=bins)
        try:
            while True:
                values = update(session, placeholder, outputs, lines, values, shape)
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
