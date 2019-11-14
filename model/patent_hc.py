#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Patent-hierarchical-classification -> patent_hc
@IDE    ：PyCharm
@Author ：Bei Wu
@Date   ：2019/11/7 22:02
@Desc   ：
=================================================="""
import os
import time

import numpy as np
import tensorflow as tf
import tflearn
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
# from tflearn.data_utils import to_categorical
from tflearn.callbacks import Callback

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

TRAIN_SIZE = 0
VALIDATION_SIZE = 0

"""
NUM_CLASS: The number of candidate categories
DIM_TEXT: The number of input test features
DIM_CITATION: The dimension of input citation embeddings
NUM_SEC: The number of categories in section layer
NUM_SECS: The number of categories in subsection layer
DIM_FEATURES = DIM_TEXT + NUM_SEC + NUM_SUBS + DIM_CITATION
"""
NUM_CLASS = 629
DIM_TEXT = 100
DIM_CITATION = 128
NUM_SEC = 8
NUM_SUBS = 123
DIM_FEATURES = 359  # DIM_FEATURES = DIM_TEXT + DIM_CITATION + NUM_SEC + NUM_SUBS


def load_data_shuffle(data_path, pre_label_path):
    start_time = time.time()
    features = np.load(data_path).astype(dtype=np.float64)
    labels = np.load(pre_label_path).astype(dtype=np.int32)
    # labels = to_categorical(labels, NUM_CLASS)

    # Generate a validation set.
    train_data = features[:TRAIN_SIZE]
    train_label = labels[:TRAIN_SIZE]
    validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_label = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
    test_label = labels[TRAIN_SIZE + VALIDATION_SIZE:]
    elapsed_time = time.time() - start_time
    print('Dataset setting is finished! Time cost is: ', elapsed_time, 's')
    start_time = time.time()
    data_train = [(t_d, t_l) for t_d, t_l in zip(train_data, train_label)]
    np.random.shuffle(data_train)
    train_data = [t_d for t_d, t_l in data_train]
    train_label = [t_l for t_d, t_l in data_train]
    elapsed_time = time.time() - start_time
    print('Dataset shuffle is finished! Time cost is: ', elapsed_time, 's')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def model(dim_embedding, batch_s):
    # Building model
    input_layer = tflearn.input_data(shape=[None, DIM_FEATURES], name='input')
    input_text = input_layer[:, 0:DIM_TEXT]
    input_citation = input_layer[:, DIM_TEXT:DIM_TEXT + DIM_CITATION]
    input_section = input_layer[:, DIM_TEXT + DIM_CITATION:DIM_TEXT + DIM_CITATION + NUM_SEC]
    input_section = onehot2categories(input_section)
    input_subsection = input_layer[:, DIM_TEXT + DIM_CITATION + NUM_SEC:]
    input_subsection = onehot2categories(input_subsection)
    # section_embedding = tf.Variable(tf.random_normal([NUM_SEC, 128]), name='group_embedding')
    # subsection_embedding = tf.Variable(tf.random_normal([NUM_SUBS, 128]), name='group_embedding')
    textual_inf = tflearn.embedding(input_text, input_dim=142698, output_dim=dim_embedding, name='word_embedding')
    section_embedding = tflearn.embedding(input_section, input_dim=NUM_SEC, output_dim=dim_embedding, name='section_embedding')
    subsection_embedding = tflearn.embedding(input_subsection, input_dim=NUM_SUBS, output_dim=dim_embedding,
                                             name='subsection_embedding')
    class_embedding = tf.Variable(tf.random_normal([NUM_CLASS, dim_embedding]), name='class_embedding')

    textual_embedding = tflearn.lstm(textual_inf, dim_embedding, dropout=0.8, name='lstm')
    network = tf.concat([textual_embedding, input_citation], 1)
    network = tflearn.fully_connected(network, dim_embedding, activation='elu')
    network = _cat_weighted(network, section_embedding)
    network = _cat_weighted(network, subsection_embedding)
    network = tf.matmul(network, tf.transpose(class_embedding), name='class_weight')
    # network = tflearn.softmax(network)
    network = tflearn.sigmoid(network)
    # network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    # tflearn.objectives.binary_crossentropy(y_pred, y_true) Computes sigmoid cross entropy between y_pred (logits) and y_true (labels).
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, dtype=tf.float64, batch_size=batch_s, loss='binary_crossentropy')
    return network


def loss_multilabel(self, l2_lambda=0.0001):  # 0.0001 this loss function is for multi-label classification
    with tf.name_scope("loss"):
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        # input_y:shape=(?, 1999); logits:shape=(?, 1999)
        # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                         logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
        # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
        print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
        losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
        loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
    return loss


def onehot2categories(arr):
    temp = []
    for i in range(len(arr)):
        print(arr[i])
        temp2 = []
        for j, value in enumerate(arr[i]):
            if value == 1:
                temp2.append(j)
        temp.append(temp2)
        return temp


def output(arr):
    temp = []
    for i in range(len(arr)):
        # print(arr[i])
        temp2 = []
        for j in range(len(arr[i])):
            if arr[i][j] > 0.5:
                temp2.append(1)
            else:
                temp2.append(0)
        temp.append(temp2)
    return temp


def _cat_weighted(patent, category):
    weight = tf.matmul(patent, tf.transpose(category))
    # probability = tf.nn.softmax(weight)  # 加一个softmax
    probability = tf.nn.sigmoid(weight)  # 加sigmoid，将所有值映射到0-1之间
    new_patent = tf.matmul(probability, category)
    return new_patent


def print_evaluation_scores(test_labels_trans, test_predict_trans):
    accuracy = accuracy_score(test_labels_trans, test_predict_trans)
    f1_score_macro = f1_score(test_labels_trans, test_predict_trans, average='macro')
    f1_score_micro = f1_score(test_labels_trans, test_predict_trans, average='micro')
    f1_score_weighted = f1_score(test_labels_trans, test_predict_trans, average='weighted')
    hamming = hamming_loss(test_labels_trans, test_predict_trans)
    test_auc = roc_auc_score(test_labels_trans, test_predict_trans, average='micro')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)
    print("hamming_loss:", hamming)
    print("test_auc: ", test_auc)


# add early_stopping
def train_predict(network, x, y, val_x, val_y, model_root_path, test_x, test_y, bs, epoch_num):
    mdl = tflearn.DNN(network, tensorboard_dir=model_root_path+"/tflearn_logs/",
                      checkpoint_path=model_root_path+"/model.tfl.ckpt")
    model_file_path = model_root_path+"model.tfl"
    if os.path.isfile(model_file_path):
        mdl.load(model_file_path)
    early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.90)
    try:
        mdl.fit(x, y, validation_set=(val_x, val_y), n_epoch=epoch_num, shuffle=True,
                snapshot_epoch=True,  # Snapshot (save & evaluate) model every epoch.
                show_metric=True, batch_size=bs, callbacks=early_stopping_cb, run_id='model')
    except StopIteration:
        print("OK, stop iterate!Good!")
    # model.fit(x, y, n_epoch=num_epoch, validation_set=(val_x, val_y), shuffle=True,
    #           show_metric=True, batch_size=bs, callbacks=early_stopping_cb, run_id='penNet')  # epoch = 150
    # Save the model
    mdl.save(model_file_path)
    print('Model storage is finished')
    test_predict = mdl.predict(test_x)
    test_predict_trans = [np.argmax(one_hot) for one_hot in test_predict]
    test_labels_trans = [np.argmax(one_hot) for one_hot in test_y]
    print_evaluation_scores(test_labels_trans, test_predict_trans)
    print('Model predict is finished')


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        # Store a validation accuracy threshold, which we can compare against
        # the current validation accuracy at, say, each epoch, each batch step, etc.
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        """
        This is the final method called in trainer.py in the epoch loop.
        We can stop training and leave without losing any information with a simple exception.
        """
        # print dir(training_state)
        print("Terminating training at the end of epoch", training_state.epoch)
        if training_state.val_acc >= self.val_acc_thresh and training_state.acc_value >= self.val_acc_thresh:
            raise StopIteration

    def on_train_end(self, training_state):
        """
        Furthermore, tflearn will then immediately call this method after we terminate training,
        (or when training ends regardless). This would be a good time to store any additional
        information that tflearn doesn't store already.
        """
        print("Successfully left training! Final model accuracy:", training_state.acc_value)


if __name__ == '__main__':
    feature_path = '/home/wubei/data_100_5_npy/all_vector.npy'
    label_path = '/home/wubei/data_100_5_npy/class_vector.npy'
    model_path = '/home/wubei/model/100_5_1024_150/'  # NameSpace: "textual_dim"_"lowest word frequency"_"batch_size"_"num_epoch"
    embedding_dim = 128
    batch_size = 1024
    num_epoch = 150
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels \
        = load_data_shuffle(feature_path, label_path)
    net = model(embedding_dim, batch_size)
    train_predict(net, train_features, train_labels, validation_features, validation_labels,
                  model_path, test_features, test_labels, batch_size, num_epoch)
