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

import tensorflow as tf

def loss_fn(labels, logits):
    """A simple softmax cross entropy loss over the outputs."""
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

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

    with tf.variable_scope('model'):
        global_step = tf.train.get_or_create_global_step()
        logits = params['model'](features, is_training=is_training)

        channel_axis = 1 if params['data_format'] == 'channels_first' else 2
        probabilities = tf.nn.softmax(logits=logits, axis=channel_axis, name='probabilities')
        predictions = tf.argmax(input=logits, axis=channel_axis, output_type=tf.int32, name='predictions')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits' : logits,
            'probabilities' : probabilities,
            'predictions' : predictions
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predictions' : tf.estimator.export.PredictOutput(predictions)
            }
        )

    with tf.variable_scope('loss'):
        # tf.losses seem to expect `channels_last`
        if params['data_format'] == 'channels_first':
            logits = tf.transpose(logits, [0, 2, 1], name='channels_last_logits')

        receptive_field = params['receptive_field']
        loss = loss_fn(labels=labels[:,receptive_field:,...], logits=logits[:,receptive_field:,...])

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
