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

from data.quantization import quantiles, quantize, dequantize

def get_numpy_data(numpy_filename, number_of_bins):
    """."""
    y = np.load(numpy_filename)

    mean = y.mean()
    std = y.std()

    y = (y-mean)/std

    # Find a roughly even quantization
    bins = quantiles(y, number_of_bins)

    # Digitize
    data = quantize(y[:-1], bins)
    data_labels = quantize(y[1:], bins, dtype=np.int32)

    # Turn feature data into sample points again
    data = dequantize(data, bins)

    return data, data_labels, bins

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print('usage: {} numpy_filename'.format(sys.argv[0]))
        exit(1)

    data, _, bins = get_numpy_data(numpy_filename=sys.argv[1], number_of_bins=1024)
    print(bins)
    print(data.shape)
    plt.plot(data)
    plt.grid(True)
    plt.show()
