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
"""Routines for dealing with quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def quantiles(data, number_of_bins, axis=None):
    """."""
    bins = np.percentile(data, np.linspace(0, 100, 1+number_of_bins), axis=axis, interpolation='linear')
    return bins.astype(np.float32)

def quantize(data, bins, right=False, dtype=np.int64):
    """."""
    quantized = np.digitize(data, bins[1:-1], right=right)
    return quantized.astype(dtype)

def dequantize(digits, bins):
    """."""
    return bins[digits]

def requantize(data, bins):
    """."""
    return dequantize(quantize(data, bins), bins)
