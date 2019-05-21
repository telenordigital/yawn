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
"""Test data in the form of a quantized sine wave with added noise."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from data.quantization import quantiles, quantize, dequantize

def get_numpy_data(dataset_size, number_of_bins, scale):
    """."""
    limits = 2.0*np.pi*scale

    x = np.linspace(-limits, limits, dataset_size+1)
    y = np.sin(x)

    # Find a roughly even quantization
    bins = quantiles(y, number_of_bins)

    # Add noise
    locs = np.array([-0.2, 0.2])
    scales = np.ones(locs.size)/1e1
    coeffs = np.ones(locs.size)/2.0

    indices = np.random.multinomial(1, coeffs, size=y.size).argmax(axis=-1)
    y += np.random.normal(loc=locs[indices], scale=scales[indices])

    # Digitize
    data = quantize(y[:-1], bins)
    data_labels = quantize(y[1:], bins, dtype=np.int32)

    # Turn feature data into sample points again
    data = dequantize(data, bins)

    return data, data_labels, bins

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data, _, _ = get_numpy_data(1000, 64, 2)
    plt.plot(data)
    plt.grid(True)
    plt.show()
