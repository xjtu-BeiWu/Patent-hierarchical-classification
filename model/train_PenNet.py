#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Patent-hierarchical-classification -> train_PenNet
@IDE    ：PyCharm
@Author ：Bei Wu
@Date   ：2019/11/18 21:37
@Desc   ：
=================================================="""
import sys
import logging
import time

from utils import data_helpers as dh


import tensorflow as tf

# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("✘ The format of your input is illegal, please re-input: ")
logging.info("✔︎ The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()))

feature_path = '/home/wubei/data_100_5_npy/all_vector.npy'
label_path = '/home/wubei/data_100_5_npy/class_vector.npy'
model_path = '/home/wubei/model/100_5_1024_150/'  # NameSpace: "textual_dim"_"lowest word frequency"_"batch_size"_"num_epoch"

# Data Parameters
tf.flags.DEFINE_string("feature_label_file", feature_path, "Data source for the features.")
tf.flags.DEFINE_string("label_data_file", label_path, "Data source for the labels.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("lstm_hidden_size", 128, "Hidden size for bi-lstm layer(default: 256)")
tf.flags.DEFINE_integer("fc_hidden_size", 512, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("num_classes_list", "8,123,629", "Number of labels list (depends on the task)")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 5000)")
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store (default: 50)")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))