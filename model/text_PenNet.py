#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Patent-hierarchical-classification -> text_PenNet
@IDE    ：PyCharm
@Author ：Bei Wu
@Date   ：2019/11/18 20:22
@Desc   ：
=================================================="""
import tensorflow as tf
import tflearn


class PenNet2(object):
    def __init__(
            self, DIM_FEATURES, DIM_TEXT, DIM_CITATION, NUM_SEC, NUM_SUBS, NUM_CLASS, vocab_size, batch_size,
            lstm_hidden_size, fc_hidden_size, embedding_size, l2_reg_lambda=0.0, alpha=0.0, pretrained_embedding=None):
        self.DIM_FEATURES = DIM_FEATURES
        self.DIM_TEXT = DIM_TEXT
        self.DIM_CITATION = DIM_CITATION
        self.NUM_SEC = NUM_SEC
        self.NUM_SUBS = NUM_SUBS
        self.NUM_CLASS = NUM_CLASS
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda
        self.alpha = alpha
        self.pretrained_embedding = pretrained_embedding

        # self.input_features = tf.placeholder(tf.int32, [None, DIM_FEATURES], name="input_features")
        self.input_text = tf.placeholder(tf.int32, [None, DIM_TEXT], name="input_text")
        self.input_citation = tf.placeholder(tf.float64, [None, DIM_CITATION], name="input_citation")
        self.input_section = tf.placeholder(tf.int32, [None, NUM_SEC], name="input_section")
        self.input_subsection = tf.placeholder(tf.int32, [None, NUM_SUBS], name="input_subsection")
        self.input_class = tf.placeholder(tf.int32, [None, NUM_CLASS], name="input_class")

        self.weight_section = tf.placeholder(tf.float64, [None, NUM_SEC], name="weight_section")
        self.weight_subsection = tf.placeholder(tf.float64, [None, NUM_SUBS], name="weight_subsection")
        self.weight_class = tf.placeholder(tf.float64, [None, NUM_CLASS], name="weight_class")

    def _cat_weighted(self, patent, category):
        weight = tf.matmul(patent, tf.transpose(category))
        # probability = tf.nn.softmax(weight)  # 加一个softmax
        probability = tf.nn.sigmoid(weight)  # 加sigmoid，将所有值映射到0-1之间
        new_patent = tf.matmul(probability, category)
        return weight, new_patent

    def model(self):
        # Building model
        # input_layer = tflearn.input_data(shape=[None, self.DIM_FEATURES], dtype=tf.float64, name='input')
        # input_text = input_layer[:, 0:self.DIM_TEXT]
        # input_citation = input_layer[:, self.DIM_TEXT:self.DIM_TEXT + self.DIM_CITATION]
        # self.input_section = input_layer[:,
        #                      self.DIM_TEXT + self.DIM_CITATION:self.DIM_TEXT + self.DIM_CITATION + self.NUM_SEC]
        # self.input_subsection = input_layer[:, self.DIM_TEXT + self.DIM_CITATION + self.NUM_SEC:]
        textual_inf = tflearn.embedding(self.input_text, input_dim=self.vocab_size, output_dim=self.embedding_size,
                                        name='word_embedding')
        section_embedding = tf.Variable(tf.random_normal([self.NUM_SEC, self.embedding_size]), name='group_embedding')
        subsection_embedding = tf.Variable(tf.random_normal([self.NUM_SUBS, self.embedding_size]),
                                           name='group_embedding')
        # section_embedding = tflearn.embedding(input_section, input_dim=NUM_SEC, output_dim=dim_embedding, name='section_embedding')
        # subsection_embedding = tflearn.embedding(input_subsection, input_dim=NUM_SUBS, output_dim=dim_embedding,
        #                                          name='subsection_embedding')
        class_embedding = tf.Variable(tf.random_normal([self.NUM_CLASS, self.embedding_size]), name='class_embedding')

        textual_embedding = tflearn.lstm(textual_inf, self.lstm_hidden_size, dropout=0.8, name='lstm')
        network = tf.concat([textual_embedding, self.input_citation], 1)
        network = tflearn.fully_connected(network, self.embedding_size, activation='elu')
        self.weight_section, network = self._cat_weighted(network, section_embedding)
        self.weight_subsection, network = self._cat_weighted(network, subsection_embedding)
        self.weight_class = tf.matmul(network, tf.transpose(class_embedding), name='class_weight')
        network = tflearn.sigmoid(self.weight_class)
        return network

    def loss_hierarchical_multilabel(self):  # this loss function is for multi-label classification
        def loss_multilabel(labels, logits):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, NUM); logits:shape=(?, NUM)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))  # shape=().   average loss in the batch
            return loss

        loss_section = loss_multilabel(self.input_section, self.weight_section)
        loss_subsection = loss_multilabel(self.input_subsection, self.weight_subsection)
        loss_class = loss_multilabel(self.input_class, self.weight_class)
        losses = tf.add_n(loss_section, loss_subsection, loss_class)
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(v, tf.float64) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
        loss = losses + l2_loss
        return loss

