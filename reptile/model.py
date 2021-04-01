"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf


OPTIMIZERS = {
    'adam': partial(tf.train.AdamOptimizer, beta1=0), # TODO: what does the partial here do?
    'sgd': tf.train.GradientDescentOptimizer
}

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self,
                num_classes,
                optimizer,
                learning_rate,
                ):
        
        optimizer = OPTIMIZERS[optimizer]

        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))

        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)

        self.minimize_op = optimizer(learning_rate).minimize(self.loss)
