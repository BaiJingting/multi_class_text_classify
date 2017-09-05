# -*- coding:utf-8 -*-
"""
Created on 2017年8月23日

@author: baijingting
"""

import logging
import os
import shutil

import numpy as np
import tensorflow as tf

from process import data_helper as dh
from process.text_cnn_rnn import TextCNNRNN

logging.getLogger().setLevel(logging.INFO)

predict_file = "data/手机百度android_离群预测.txt"


def map_word_to_index(examples, words_index):
    """

    :param examples:
    :param words_index:
    :return:
    """
    x_ = []
    for example in examples:
        temp = []
        for word in example:
            if word in words_index:
                temp.append(words_index[word])
            else:
                temp.append(0)
        x_.append(temp)
    return x_


def predict_unseen_data():
    """

    :return:
    """
    trained_dir = 'trained_results/'
    test_file = "data/手机百度android_离群集.txt"

    ret = dh.load_trained_params(trained_dir)
    params, words_index, labels, embedding_mat = ret[0], ret[1], ret[2], ret[3]
    raw_data, data = dh.load_test_data(test_file)
    data = dh.pad_sentences(data, forced_sequence_length=params['sequence_length'])
    data = map_word_to_index(data, words_index)
    x_test = np.asarray(data)

    predicted_dir = './predicted_results' + '/'
    if os.path.exists(predicted_dir):
        shutil.rmtree(predicted_dir)
    os.makedirs(predicted_dir)

    tf.reset_default_graph()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
                embedding_mat=embedding_mat,
                non_static=params['non_static'],
                hidden_unit=params['hidden_unit'],
                sequence_length=len(x_test[0]),
                max_pool_size=params['max_pool_size'],
                filter_sizes=map(int, params['filter_sizes'].split(",")),
                num_filters=params['num_filters'],
                num_classes=len(labels),
                embedding_size=params['embedding_dim'],
                l2_reg_lambda=params['l2_reg_lambda'])

            def real_len(batches):
                """

                :param batches:
                :return:
                """
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size'])
                        for batch in batches]

            def predict_step(x_batch):
                """

                :param x_batch:
                :return:
                """
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.dropout_keep_prob: 1.0,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                predictions = sess.run([cnn_rnn.predictions], feed_dict)
                return predictions

            checkpoint_file = trained_dir + "best_model.ckpt"

            saver = tf.train.Saver(tf.global_variables())
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file[:-5]))
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_file)
            logging.critical('{} has been loaded'.format(checkpoint_file))

            batches = dh.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

            predictions, predict_labels = [], []
            for x_batch in batches:
                batch_predictions = predict_step(x_batch)[0]
                for batch_prediction in batch_predictions:
                    predictions.append(batch_prediction)
                    predict_labels.append(labels[batch_prediction])

            with open(predict_file, 'w') as f:
                for i in range(len(raw_data)):
                    f.write(raw_data[1])
                    f.write('\t')
                    f.write(predict_labels[i])
                    f.write('\n')

            logging.critical('Prediction is complete, all files have been saved: {}'
                             .format(predicted_dir))


if __name__ == '__main__':
    predict_unseen_data()
