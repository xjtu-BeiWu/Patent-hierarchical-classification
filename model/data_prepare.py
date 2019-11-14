#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Patent-hierarchical-classification -> data_prepare
@IDE    ：PyCharm
@Author ：Bei Wu
@Date   ：2019/11/7 22:04
@Desc   ：
=================================================="""
import time

import numpy as np


def extract_data(filename, num_patents, d_features):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    print('Extracting', filename)
    data = np.loadtxt(filename)  # 从文件读取数据，存为numpy数组
    data = np.frombuffer(data).astype(np.float64)  # 改变数组元素变为float32类型
    data = data.reshape(num_patents, d_features)  # 所有元素
    return data


def extract_labels(filename, num_patents, d_label):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    label = np.loadtxt(filename)
    label = np.frombuffer(label).astype(np.int32)
    label = label.reshape(num_patents, d_label)  # 标签
    # labels = labels.flatten()
    return label


if __name__ == '__main__':
    input_file_path = '/home/wubei/data_100_5/all_vector.txt'
    input_label_path = '/home/wubei/data_100_5/class_vector.txt'
    output_file_path = '/home/wubei/data_100_5_npy/all_vector.npy'
    output_label_path = '/home/wubei/data_100_5_npy/class_vector.npy'
    num_patent = 1807188
    dim_feature = 359
    dim_label = 629
    start_time = time.time()
    print("Start transfer data!")
    feature = extract_data(input_file_path, num_patent, dim_feature)
    feature_file = np.save(output_file_path, feature)
    labels = extract_labels(input_label_path, num_patent, dim_label)
    label_file = np.save(output_label_path, labels)
    elapsed_time = time.time() - start_time
    print(elapsed_time, 's')
