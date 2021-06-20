# Code adapted from 
# 1. https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
# 2. https://github.com/YangZhang4065/AdaptationSeg/blob/master/FCN_da.py

import numpy as np

import tensorflow as tf


def weighted_ce_loss(num_classes = 20, class_to_ignore = 0):
    mask = np.ones(num_classes)
    mask[class_to_ignore] = 0
    mask = tf.keras.backend.variable(mask, dtype='float32')

    def wce_loss(y_true, y_pred, from_logits=True):
        # Preprocess data
        if from_logits == True:
            y_pred = tf.keras.backend.softmax(y_pred, axis = -1)

        # See https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1)

        loss = tf.keras.backend.categorical_crossentropy(y_true * mask, y_pred, axis=-1)

        return tf.keras.backend.mean(loss)

    return wce_loss


def masked_ce_loss(num_classes = 20, class_to_ignore = 0, from_logits=False):
    mask = np.ones(num_classes)
    mask[class_to_ignore] = 0
    mask = tf.keras.backend.variable(mask, dtype='float32')

    def masked_loss(y_true, y_pred):
        # Preprocess data
        if from_logits == True:
            y_pred = tf.keras.backend.softmax(y_pred, axis = -1)
        
        # See https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1)

        loss = tf.keras.backend.categorical_crossentropy(y_true * mask, y_pred, axis=-1)

        return tf.keras.backend.mean(loss)

    return masked_loss
