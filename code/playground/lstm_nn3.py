import pandas as pd
import numpy as np

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import sys
sys.path.append('../Util/')
from early_stopping import EarlyStopping
from SNLI import SNLI
from functions4tf import *

if __name__ == '__main__':

    maxlen = 78

    nlp_path = '/media/hirotoshi/BigData/ml/NLP/'
    snli_data_path = nlp_path + '/SNLI/snli_1.0/'
    embd_data_path = nlp_path + '/word_embed/'

    model_path = './'

    # train_df = pd.read_csv(snli_data_path + "/snli_1.0_train.txt", sep="\t", header=0)
    # dev_df   = pd.read_csv(snli_data_path + "/snli_1.0_dev.txt", sep="\t", header=0)
    # test_df  = pd.read_csv(snli_data_path + "/snli_1.0_test.txt", sep="\t", header=0)
    #
    # snli = snli(train_df=train_df, dev_df=dev_df, test_df=dev_df)

    # snliをserialize
    # import pickle
    # with open('snli.pickle', mode='wb') as f:
    #     pickle.dump(snli, f)

    # ==============================================
    # create embedding_matrix and save (if needed)
    # ==============================================
    # embd_data = embd_data_path + '/glove.6b.300d.txt'
    # create_embedding_matrix(embd_data, snli.tokenizer.word_index, embedding_dim=300,
    #                         output_file_name=nlp_path + '/snli/embedding_matrix/snli_glove.6b.300d.npy')

    embd_data_npy = nlp_path + '/SNLI/embedding_matrix/snli_glove.6B.300d.npy'

    snli_pickle_path = nlp_path + '/SNLI/pickles/snli.pickle'
    import pickle

    with open(snli_pickle_path, mode='rb') as f:
        snli = pickle.load(f)

    embedding_dim = 300
    batch_size = 512
    classes = 3

    epochs = 20

    pretrained_weights = np.load(embd_data_npy)

    x_train_premise = snli.x_train[0]
    x_train_hypothesis = snli.x_train[1]
    y_train = snli.y_train

    train_size = x_train_premise.shape[0]
    n_batches = train_size // batch_size

    x_dev_premise = snli.x_dev[0]
    x_dev_hypothesis = snli.x_dev[1]
    y_dev = snli.y_dev

    dev_size = x_dev_premise.shape[0]
    n_batches_dev = dev_size // batch_size

    x_test_premise = snli.x_test[0]
    x_test_hypothesis = snli.x_train[1]
    y_test = snli.y_test
    test_size = x_test_premise.shape[0]
    n_batches_test = test_size // batch_size


    # placeholder群
    x_premise      = tf.placeholder(tf.int32, [None, maxlen])
    x_hypothesis   = tf.placeholder(tf.int32, [None, maxlen])
    y              = tf.placeholder(tf.float32, [None, classes])
    embedding_arr  = tf.placeholder(tf.float32, [34607, embedding_dim])
    keep_prob      = tf.placeholder(tf.float32)
    keep_prob_lstm = tf.placeholder(tf.float32)
    is_training    = tf.placeholder(tf.bool)

    # embedding
    with tf.name_scope("embedding"):
        x_pre_embed = tf.nn.embedding_lookup(embedding_arr, x_premise)
        x_hyp_embed = tf.nn.embedding_lookup(embedding_arr, x_hypothesis)

    # lstm
    with tf.name_scope("lstm_1"):
        with tf.variable_scope("lstm_1", reuse=tf.AUTO_REUSE):
            lstm_units = 200
            cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
            lstm_out_pre, _ = tf.nn.dynamic_rnn(cell=cell_1, inputs=x_pre_embed, dtype=tf.float32)
            last_out_pre = lstm_out_pre[:, -1, :]


    with tf.name_scope("lstm_2"):
        with tf.variable_scope("lstm_2", reuse=tf.AUTO_REUSE):
            lstm_units = 200
            cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
            lstm_out_hyp, _ = tf.nn.dynamic_rnn(cell=cell_2, inputs=x_hyp_embed, dtype=tf.float32)
            last_out_hyp = lstm_out_hyp[:, -1, :]

    lstm_last_out = tf.concat([last_out_pre, last_out_hyp], axis=1)
    lstm_last_out = tf.nn.dropout(lstm_last_out, keep_prob_lstm)

    # 全結合nn層
    with tf.name_scope("nn1"):
        nn_units = 300
        w1  = tf.Variable(tf.truncated_normal(([lstm_units*2, nn_units]), stddev=0.1), name='w1')
        h1  = tf.Variable(tf.zeros([nn_units]), name='h1')
        alpha1 = tf.Variable(tf.zeros([nn_units]))
        out1 = tf.matmul(lstm_last_out, w1) + h1
        out1 = batch_norm_wrapper(out1, is_training)
        out1 = prelu(out1, alpha1)
        out1 = tf.nn.dropout(out1, keep_prob)
        # out1 = tf.nn.relu(out1)
        # out1 = tf.nn.softmax(tf.matmul(lstm_last_out, w1) + h1)

    with tf.name_scope("nn2"):
        w2 = tf.Variable(tf.truncated_normal(([nn_units, nn_units]), stddev=0.1))
        h2 = tf.Variable(tf.zeros([nn_units]))
        alpha2 = tf.Variable(tf.zeros([nn_units]))
        out2 = tf.matmul(out1, w2) + h2
        out2 = batch_norm_wrapper(out2, is_training)
        out2 = prelu(out2, alpha2)
        out2 = tf.nn.dropout(out2, keep_prob)
    # out2 = tf.nn.relu(out2)
    # out2 = tf.nn.softmax(tf.matmul(out1, w2) + h2)
    # out2 = batch_norm_wrapper(out2, is_training)

    with tf.name_scope("nn3"):
        w3 = tf.Variable(tf.truncated_normal(([nn_units, nn_units]), stddev=0.1))
        h3 = tf.Variable(tf.zeros([nn_units]))
        alpha3 = tf.Variable(tf.zeros([nn_units]))
        af3 = tf.matmul(out2, w3) + h3
        out3 = batch_norm_wrapper(af3, is_training)
        out3 = prelu(out2, alpha3)

    # 出力層
    with tf.name_scope("inference"):
        v = tf.Variable(tf.truncated_normal(([nn_units, classes]), stddev=0.1))
        b = tf.Variable(tf.zeros([classes]))

        out = tf.nn.softmax(tf.matmul(out3, v) + b)
        # out = tf.nn.sigmoid(tf.matmul(out3, v) + b)

    # 誤差
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(out * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                                                  reduction_indices=[1]))

    # 訓練
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0008, beta1=0.9, beta2=0.999)
    train_step = optimizer.minimize(cross_entropy)

    # 評価
    correct_prediction = tf.equal(tf.argmax(y, -1), tf.arg_max(out, -1))
    prediction_result  = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(prediction_result)

    # sessionの準備
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # sess = tf_debug.localclidebugwrappersession(sess)

    summary_writer = tf.summary.FileWriter("../../logs/playground/lstm_nn3", sess.graph)

    es = EarlyStopping(mode='acc', patience=3, verbose=1)

    for epoch in range(epochs):

        for i in range(n_batches):
            if i % 10 == 0:
                print("batch: {}/{}".format(i, n_batches))
            start = i * batch_size
            end = start + batch_size

            print(sess.run(cross_entropy, feed_dict={
                x_premise: x_train_premise[start:end],
                x_hypothesis: x_train_hypothesis[start:end],
                embedding_arr: pretrained_weights,
                y: y_train[start:end],
                keep_prob: 0.5,
                keep_prob_lstm: 0.5,
                is_training: False
            }))
            sess.run(train_step, feed_dict={
                x_premise: x_train_premise[start:end],
                x_hypothesis: x_train_hypothesis[start:end],
                embedding_arr: pretrained_weights,
                y: y_train[start:end],
                keep_prob: 0.5,
                keep_prob_lstm: 0.5,
                is_training: True
            })

        print("evaluating by using dev data....")
        dev_pre_results = []
        for i in range(n_batches_dev):

            res = prediction_result.eval(session=sess, feed_dict={
                x_premise: x_dev_premise,
                x_hypothesis: x_dev_hypothesis,
                embedding_arr: pretrained_weights,
                y: y_dev,
                keep_prob: 1.0,
                keep_prob_lstm: 1.0,
                is_training: False
            })
            dev_pre_results.append(res)
        acc = np.mean(dev_pre_results)


        print('epoch: ', epoch,
              # 'dev_pre_results', dev_pre_results,
              'dev_acc', acc)

        if es.validate(acc):
            break


    # testデータでの評価
    test_pre_results = []
    for i in range(n_batches_dev):

        res = prediction_result.eval(session=sess, feed_dict={
            x_premise: x_test_premise,
            x_hypothesis: x_test_hypothesis,
            embedding_arr: pretrained_weights,
            y: y_dev,
            keep_prob: 1.0,
            keep_prob_lstm: 1.0,
            is_training: False
        })
        test_pre_results.append(res)
    acc = np.mean(test_pre_results)


    print('epoch: ', epoch,
          # 'dev_pre_results', dev_pre_results,
          'test_acc', acc)
#    in_dim, out_dim = pretrained_weights.shape
