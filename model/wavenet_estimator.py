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
"""Glue between our WaveNet model and the tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# TODO: These loss functions belong in some other file?
def regressive_loss_fn(labels, predictions, params):
    """."""
    labels = tf.gather(predictions['bins'], labels)

    return tf.losses.mean_squared_error(
        labels[:,params['receptive_field']:,...],
        predictions['values'][:,params['receptive_field']:,...]
    )

def categorical_loss_fn(labels, predictions, params):
    """."""
    logits = predictions['logits']

    # tf.losses seem to expect `channels_last`
    if params['data_format'] == 'channels_first':
        logits = tf.transpose(logits, [0, 2, 1], name='channels_last_logits')

    return tf.losses.sparse_softmax_cross_entropy(
        labels=labels[:,params['receptive_field']:,...],
        logits=logits[:,params['receptive_field']:,...]
    )

def mixture_loss_normalize(value, predictions):
    """."""
    return (value - predictions['means'])*predictions['inverse_standard_deviations']

def mixture_loss_cdf(value, predictions):
    """."""
    normalized_value = mixture_loss_normalize(value, predictions)
    return 0.5*(1.0+tf.erf(normalized_value/np.sqrt(2.0)))

def mixture_loss_fn(labels, predictions, params):
    """."""
    labels = tf.tile(labels, [1, 1, params['num_mixtures']])

    lower_bin = tf.gather(predictions['bins'], labels)
    upper_bin = tf.gather(predictions['bins'], labels+1)

    lower_cdf = mixture_loss_cdf(lower_bin, predictions)
    upper_cdf = mixture_loss_cdf(upper_bin, predictions)

    # -infinity is our lowest bin
    lower_cdf = tf.where(
        tf.equal(labels, 0),
        tf.zeros_like(lower_cdf),
        lower_cdf
    )

    # +infinity is our highest bin
    upper_cdf = tf.where(
        tf.equal(labels, params['quantization']-1),
        tf.ones_like(upper_cdf),
        upper_cdf
    )

    label_probabilities = upper_cdf - lower_cdf
    label_probabilities = tf.reduce_mean(label_probabilities, axis=-1)

    # Finally maximize the log probability of the observed bins
    return tf.reduce_mean(-tf.log(label_probabilities))

LOSS_FNS = {
    'regressive'  : regressive_loss_fn,
    'categorical' : categorical_loss_fn,
    'mixture'     : mixture_loss_fn
}

def train_fn(loss, global_step, learning_rate):
    """A simple adam optimizer applied to the loss."""
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def model_fn(features, labels, mode, params):
    """Model function for WaveNet.
    Arguments:
      features: This is the first item returned from the `input_fn`
        passed to `train`, `evaluate`, and `predict`. This should be a
        single `Tensor` or `dict` of same.
      labels: This is the second item returned from the `input_fn`
        passed to `train`, `evaluate`, and `predict`. This should be a
        single `Tensor` or `dict` of same (for multi-head models). If
        mode is `ModeKeys.PREDICT`, `labels=None` will be passed. If
        the `model_fn`'s signature does not accept `mode`, the
        `model_fn` must still be able to handle `labels=None`.
      mode: Optional. Specifies if this training, evaluation or
        prediction. See `ModeKeys`.
      params: Optional `dict` of hyperparameters.  Will receive what
        is passed to Estimator in `params` parameter. This allows
        to configure Estimators from hyper parameter tuning.
    Returns:
      A tf.estimator.EstimatorSpec for the given mode.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    channel_axis = 1 if params['data_format'] == 'channels_first' else 2

    with tf.variable_scope('model'):
        global_step = tf.train.get_or_create_global_step()
        predictions = params['model'](features, is_training=is_training)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predictions' : tf.estimator.export.PredictOutput(predictions)
            }
        )

    with tf.variable_scope('loss'):
        loss_fn = LOSS_FNS[params['version']]
        loss = loss_fn(labels=labels, predictions=predictions, params=params)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )

    with tf.variable_scope('train'):
        train_op = train_fn(loss, global_step, learning_rate=params['learning_rate'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )
