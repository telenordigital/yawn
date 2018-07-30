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
"""Test data in the form of a quantized sine wave."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def quantize(data, bins, right=False, dtype=np.int64):
    """."""
    quantized = np.digitize(data, bins[1:-1], right=right)
    return quantized.astype(dtype)

def dequantize(digits, bins):
    """."""
    return bins[digits]

def get_numpy_data(dataset_size, number_of_bins, scale):
    """."""
    limits = 2.0*np.pi*scale

    x = np.linspace(-limits, limits, dataset_size+1)
    y = np.sin(x)

    # Find a roughly even quantization
    bins = np.percentile(y, np.linspace(0, 100, 1+number_of_bins), interpolation='linear')
    bins = bins.astype(np.float32)

    # Add noise
    y += np.random.uniform(low=-0.2, high=0.2, size=y.shape)

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
