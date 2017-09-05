# -*- coding:utf-8 -*-
"""
Created on 2017年8月23日

@author: baijingting
"""

import jieba
import json
import pickle
import logging
import itertools
import numpy as np
from collections import Counter

logging.getLogger().setLevel(logging.INFO)


def load_train_data(inputfile, outlier_data):
    """

    :param inputfile:
    :param outlier_data:
    :return:
    """
    train_data = []
    train_label = []
    test_data = []
    test_raw_data = []
    labels = set()
    with open(outlier_data, 'w') as f1:
        with open(inputfile, 'r') as f2:
            for line in f2.readlines():
                line = line.strip('\n').split(',')
                raw_content = ""
                for i in range(len(line) - 1):
                    raw_content += line[i]
                if line[-1] == '-1':
                    f1.write(raw_content)
                    f1.write('\n')
                    content = clean_str(raw_content)
                    if len(content) == 0:
                        continue
                    test_raw_data.append(raw_content)
                    content = list(jieba.cut(content))
                    test_data.append(content)
                else:
                    content = clean_str(raw_content)
                    if len(content) == 0:
                        continue
                    content = list(jieba.cut(content))
                    train_data.append(content)
                    train_label.append(int(line[-1]))
                    labels.add(int(line[-1]))

    labels = sorted(list(labels))
    train = [train_data, train_label]
    test = [test_raw_data, test_data]
    return train, test, labels


def produce_label_dict(labels):
    """

    :param labels:
    :return:
    """
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    return label_dict


def clean_str(s):
    """

    :param s:
    :return:
    """
    s = s.replace(' ', '').replace('，', '').replace('。', '') \
        .replace('！', '').replace('?', '').replace('_', '')
    return s.strip('\n')


def load_embeddings(vocabulary):
    """

    :param vocabulary:
    :return:
    """
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """
    Pad setences during training or prediction
    :param sentences:
    :param padding_word:
    :param forced_sequence_length:
    :return:
    """
    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:  # Prediction
        logging.critical('Prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:
            # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off '
                         'because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    """

    :param sentences:
    :return:
    """
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """

    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def real_len(batches, params):
    """

    :param batches:
    :param params:
    :return:
    """
    return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]


def train_step(x_batch, y_batch, params, cnn_rnn, sess, train_op, global_step):
    """

    :param x_batch:
    :param y_batch:
    :param params:
    :param cnn_rnn:
    :param sess:
    :param train_op:
    :param global_step:
    :return:
    """
    feed_dict = {
        cnn_rnn.input_x: x_batch,
        cnn_rnn.input_y: y_batch,
        cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
        cnn_rnn.batch_size: len(x_batch),
        cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
        cnn_rnn.real_len: real_len(x_batch, params),
    }
    _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy],
                                       feed_dict)


def dev_step(x_batch, y_batch, params, cnn_rnn, sess, global_step):
    """

    :param x_batch:
    :param y_batch:
    :param params:
    :param cnn_rnn:
    :param sess:
    :param global_step:
    :return:
    """
    feed_dict = {
        cnn_rnn.input_x: x_batch,
        cnn_rnn.input_y: y_batch,
        cnn_rnn.dropout_keep_prob: 1.0,
        cnn_rnn.batch_size: len(x_batch),
        cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
        cnn_rnn.real_len: real_len(x_batch, params),
    }
    step, loss, accuracy, num_correct, predictions = sess.run(
        [global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions],
        feed_dict)
    ret = [accuracy, loss, num_correct, predictions]
    return ret


def load_test_data(test_file):
    """

    :param test_file:
    :return:
    """
    raw_data = []
    data = []
    with open(test_file, 'r') as f:
        for line in f.readlines():
            raw_data.append(line)
            line = clean_str(line)
            line = list(jieba.cut(line))
            data.append(line)
    return raw_data, data


def load_trained_params(trained_dir):
    """

    :param trained_dir:
    :return:
    """
    params = json.loads(open(trained_dir + 'trained_parameters.json').read())
    words_index = json.loads(open(trained_dir + 'words_index.json').read())
    labels = json.loads(open(trained_dir + 'labels.json').read())

    with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
        fetched_embedding = pickle.load(input_file)
    embedding_mat = np.array(fetched_embedding, dtype=np.float32)
    ret = [params, words_index, labels, embedding_mat]
    return ret


def real_len(batches, params):
    """

    :param batches:
    :param params:
    :return:
    """
    return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size'])
            for batch in batches]


def predict_step(x_batch, sess, cnn_rnn, params):
    """

    :param x_batch:
    :param sess:
    :param cnn_rnn:
    :param params:
    :return:
    """
    feed_dict = {
        cnn_rnn.input_x: x_batch,
        cnn_rnn.dropout_keep_prob: 1.0,
        cnn_rnn.batch_size: len(x_batch),
        cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
        cnn_rnn.real_len: real_len(x_batch, params),
    }
    predictions = sess.run([cnn_rnn.predictions], feed_dict)
    return predictions
