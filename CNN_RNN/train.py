# -*- coding:utf-8 -*-
"""
Created on 2017年8月23日

@author: baijingting
"""

import json
import logging
import os
import pickle
import shutil
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split

from process import data_helper as dh
from process.text_cnn_rnn import TextCNNRNN

training_config = "config/training_config.json"
inputfile = "data/手机百度android_classify.txt"
outlier_data = "data/手机百度android_离群集.txt"
predict_file = "data/手机百度android_离群预测.txt"

logging.getLogger().setLevel(logging.INFO)


def train_cnn_rnn():
    """

    :return:
    """
    params = json.loads(open(training_config).read())

    train, test, labels = dh.load_train_data(inputfile, outlier_data)
    train_data = train[0]
    train_label = train[1]
    test_raw_data = test[0]
    test_data = test[1]
    label_dict = dh.produce_label_dict(labels)

    train_len = len(train_data)
    x_raw = train_data + test_data
    x_raw = dh.pad_sentences(x_raw)
    vocabulary, vocabulary_inv = dh.build_vocab(x_raw)

    train = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])

    train_data = train[:train_len]
    test_data = train[train_len:]
    train_Y = np.array([label_dict[y] for y in train_label])

    print train_data.shape
    print train_Y.shape

    # Assign a 300 dimension vector to each word
    word_embeddings = dh.load_embeddings(vocabulary)
    embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
    embedding_mat = np.array(embedding_mat, dtype=np.float32)

    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(train_data, train_Y, test_size=0.1)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'
                 .format(len(x_train), len(x_dev), len(test_data)))

    # Create a directory, everything related to the training will be saved in this directory
    trained_dir = './trained_results' + '/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
                embedding_mat=embedding_mat,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                non_static=params['non_static'],
                hidden_unit=params['hidden_unit'],
                max_pool_size=params['max_pool_size'],
                filter_sizes=map(int, params['filter_sizes'].split(",")),
                num_filters=params['num_filters'],
                embedding_size=params['embedding_dim'],
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
            grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Checkpoint files will be saved in this directory during training
            checkpoint_dir = './checkpoints' + '/'
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = dh.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                          params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            # Train the model with x_train and y_train
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                dh.train_step(x_train_batch, y_train_batch, params, cnn_rnn,
                              sess, train_op, global_step)
                current_step = tf.train.global_step(sess, global_step)

                # Evaluate the model with x_dev and y_dev
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = dh.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        ret = dh.dev_step(x_dev_batch,
                                                y_dev_batch, params, cnn_rnn, sess, global_step)
                        acc, loss, num_dev_correct, predictions = ret[0], ret[1], ret[2], ret[3]
                        total_dev_correct += num_dev_correct
                    accuracy = float(total_dev_correct) / len(y_dev)
                    logging.info('Accuracy on dev set: {}'.format(accuracy))

                    if accuracy >= best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'
                                         .format(best_accuracy, best_at_step))
            logging.critical('Training is complete, testing the best model on x_test and y_test')

        # Save trained parameters and files since predict.py needs them
        with open(trained_dir + 'words_index.json', 'w') as outfile:
            json.dump(vocabulary, outfile, indent=4)
        with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
            pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
        with open(trained_dir + 'labels.json', 'w') as outfile:
            json.dump(labels, outfile, indent=4)

            os.rename(path + '.index', trained_dir + 'best_model.ckpt')
            os.rename(path + '.meta', trained_dir + 'best_model.meta')
        # shutil.rmtree(checkpoint_dir)

        params['sequence_length'] = x_train.shape[1]
        with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
            json.dump(params, outfile, indent=4, sort_keys=True)

        # Evaluate x_test and y_test
        saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))

        batches = dh.batch_iter(list(test_data), params['batch_size'], 1, shuffle=False)

        predictions, predict_labels = [], []
        for x_batch in batches:
            batch_predictions = dh.predict_step(x_batch, sess, cnn_rnn, params)[0]
            for batch_prediction in batch_predictions:
                predictions.append(batch_prediction)
                predict_labels.append(labels[batch_prediction])

        with open(predict_file, 'w') as f:
            for i in range(len(test_raw_data)):
                f.write(test_raw_data[i])
                f.write('\t')
                f.write(str(predict_labels[i]))
                f.write('\n')


if __name__ == '__main__':
    train_cnn_rnn()
