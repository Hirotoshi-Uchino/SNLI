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
from logger import Logger


def hist_add(hist, hist_sum, target, suffix_sum='_sum'):
    if hist_sum[target + suffix_sum] is None:
        hist_sum[target + suffix_sum] = np.zeros(np.array(hist[target]).shape)

    hist_sum[target + suffix_sum] += np.array(hist[target])

def hist_add_all(hist, hist_sum):
   for k in hist.keys():
       hist_add(hist, hist_sum, k)


if __name__ == '__main__':

    nlp_path = '/media/hirotoshi/BigData/ml/NLP/'
    snli_data_path = nlp_path + '/SNLI/snli_1.0/'
    embd_data_path = nlp_path + '/word_embed/'

    maxlen = 78
    # model_path = './'

    # train_df = pd.read_csv(snli_data_path + "/snli_1.0_train.txt", sep="\t", header=0)
    # dev_df   = pd.read_csv(snli_data_path + "/snli_1.0_dev.txt", sep="\t", header=0)
    # test_df  = pd.read_csv(snli_data_path + "/snli_1.0_test.txt", sep="\t", header=0)
    #
    # snli = SNLI(train_df=train_df, dev_df=dev_df, test_df=test_df)

    # snliをserialize
    # import pickle
    # with open('snli.pickle', mode='wb') as f:
    #      pickle.dump(snli, f)

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
    batch_size = 4096
    classes = 3

    # epochs = 1
    epochs = 100


    pretrained_weights = np.load(embd_data_npy)

    x_train_premise = snli.x_train[0]
    x_train_hypothesis = snli.x_train[1]
    y_train = snli.y_train

    train_size = x_train_premise.shape[0]
    train_batch_indexes = make_batch_indexes(train_size, batch_size)

    x_dev_premise = snli.x_dev[0]
    x_dev_hypothesis = snli.x_dev[1]
    y_dev = snli.y_dev

    dev_size = x_dev_premise.shape[0]
    dev_batch_indexes = make_batch_indexes(dev_size, batch_size)

    x_test_premise = snli.x_test[0]
    x_test_hypothesis = snli.x_test[1]
    y_test = snli.y_test

    test_size = x_test_premise.shape[0]
    test_batch_indexes = make_batch_indexes(test_size, batch_size)

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
    lstm_units = 200
    cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob_lstm)
    with tf.name_scope("lstm_1"):
        lstm_out_pre, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x_pre_embed, dtype=tf.float32)
        last_out_pre = lstm_out_pre[:, -1, :]

    with tf.name_scope('lstm_2'):
        lstm_out_hyp, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x_hyp_embed, dtype=tf.float32)
        last_out_hyp = lstm_out_hyp[:, -1, :]


    # with tf.name_scope("lstm_1"):
    #     with tf.variable_scope("lstm_1", reuse=None):
    #         lstm_units = 200
    #         cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
    #         cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell_1, output_keep_prob=keep_prob_lstm)
    #         lstm_out_pre, _ = tf.nn.dynamic_rnn(cell=cell_1, inputs=x_pre_embed, dtype=tf.float32)
    #         last_out_pre = lstm_out_pre[:, -1, :]
    #
    #
    # with tf.name_scope("lstm_2"):
    #     with tf.variable_scope("lstm_2", reuse=None):
    #         lstm_units = 200
    #         cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
    #         cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell_2, output_keep_prob=keep_prob_lstm)
    #         lstm_out_hyp, _ = tf.nn.dynamic_rnn(cell=cell_2, inputs=x_hyp_embed, dtype=tf.float32)
    #         last_out_hyp = lstm_out_hyp[:, -1, :]

    lstm_last_out = tf.concat([last_out_pre, last_out_hyp], axis=1)

    # 全結合nn層
    with tf.name_scope("nn1"):
        nn_units = 300
        w1  = tf.Variable(tf.truncated_normal(([lstm_units*2, nn_units]), stddev=0.1), name='w1')
        h1  = tf.Variable(tf.zeros([nn_units]), name='h1')
        alpha1 = tf.Variable(tf.zeros([nn_units]))
        out1 = tf.matmul(lstm_last_out, w1) + h1
        #out1 = batch_norm_wrapper(out1, is_training)
        out1 = prelu(out1, alpha1)
        out1 = tf.nn.dropout(out1, keep_prob)
        out1 = batch_norm_wrapper(out1, is_training)
        # out1 = tf.nn.relu(out1)
        # out1 = tf.nn.softmax(tf.matmul(lstm_last_out, w1) + h1)

    with tf.name_scope("nn2"):
        w2 = tf.Variable(tf.truncated_normal(([nn_units, nn_units]), stddev=0.1))
        h2 = tf.Variable(tf.zeros([nn_units]))
        alpha2 = tf.Variable(tf.zeros([nn_units]))
        out2 = tf.matmul(out1, w2) + h2
        #out2 = batch_norm_wrapper(out2, is_training)
        out2 = prelu(out2, alpha2)
        out2 = tf.nn.dropout(out2, keep_prob)
        out2 = batch_norm_wrapper(out2, is_training)
    # out2 = tf.nn.relu(out2)
    # out2 = tf.nn.softmax(tf.matmul(out1, w2) + h2)
    # out2 = batch_norm_wrapper(out2, is_training)

    with tf.name_scope("nn3"):
        w3 = tf.Variable(tf.truncated_normal(([nn_units, nn_units]), stddev=0.1))
        h3 = tf.Variable(tf.zeros([nn_units]))
        alpha3 = tf.Variable(tf.zeros([nn_units]))
        out3 = tf.matmul(out2, w3) + h3
        # out3 = batch_norm_wrapper(af3, is_training)
        out3 = prelu(out3, alpha3)
        out3 = tf.nn.dropout(out3, keep_prob)
        out3 = batch_norm_wrapper(out3, is_training)
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
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
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

    # summary_writer = tf.summary.FileWriter("../../logs/playground/lstm_drop_nn3", sess.graph)

    logger = Logger("../../logs/playground/lstm_drop_nn3", sess.graph)

    es = EarlyStopping(mode='acc', patience=3, verbose=1)

    for epoch in range(epochs):

        # shuffle
        perm = np.random.permutation(x_train_premise.shape[0])
        x_train_premise_input    = x_train_premise[perm, :]
        x_train_hypothesis_input = x_train_hypothesis[perm, :]
        y_train_input            = y_train[perm, :]

        i = 0
        train_pre_result = 0

        hist_dic = {'rw1': None, 'rw2': None, 'rw3': None,
                     'rh1': None, 'rh2': None, 'rh3': None,
                     'rv': None, 'rb': None}
        hist_sum_dic = {'rw1_sum': None, 'rw2_sum': None, 'rw3_sum': None,
                         'rh1_sum': None, 'rh2_sum': None, 'rh3_sum': None,
                         'rv_sum': None, 'rb_sum': None}

        for index in train_batch_indexes:
            start = index['start']
            end   = index['end']

            if i % 10 == 0:
                print("batch: {}/{}".format(i, len(train_batch_indexes)))

            ce, ac, pr = sess.run([cross_entropy, accuracy, prediction_result], feed_dict={
                x_premise: x_train_premise_input[start:end],
                x_hypothesis: x_train_hypothesis_input[start:end],
                embedding_arr: pretrained_weights,
                y: y_train_input[start:end],
                keep_prob: 1.0,
                keep_prob_lstm: 1.0,
                is_training: False
            })

            rw1, rw2, rw3, rh1, rh2, rh3, rv, rb =\
                sess.run([w1, w2, w3, h1, h2, h3, v, b], feed_dict={
                x_premise: x_train_premise_input[start:end],
                x_hypothesis: x_train_hypothesis_input[start:end],
                embedding_arr: pretrained_weights,
                y: y_train_input[start:end],
                keep_prob: 1.0,
                keep_prob_lstm: 1.0,
                is_training: False
            })

            print('batch cross entropy: ', ce, '  batch accuracy: ', ac)
            # print(rw1)
            train_pre_result += np.sum(pr)

            sess.run(train_step, feed_dict={
                x_premise: x_train_premise_input[start:end],
                x_hypothesis: x_train_hypothesis_input[start:end],
                embedding_arr: pretrained_weights,
                y: y_train_input[start:end],
                keep_prob: 1.0,
                keep_prob_lstm: 1.0,
                is_training: True
            })

            # if 'rw1_sum' in locals():
            #     rw1_sum += np.array(rw1)
            #
            # else:
            #     rw1_sum = np.zeros(np.array(rw1).shape)

            hist_dic['rw1'] = rw1
            hist_dic['rw2'] = rw2
            hist_dic['rw3'] = rw3
            hist_dic['rh1'] = rh1
            hist_dic['rh2'] = rh2
            hist_dic['rh3'] = rh3
            hist_dic['rv']  = rv
            hist_dic['rb']  = rb
            hist_add_all(hist_dic, hist_sum_dic)

            i += 1

        train_acc = train_pre_result / train_size
        # rw1_ave   = rw1_sum / train_size

        print("evaluating by using dev data....")
        dev_pre_result = 0
        for index in dev_batch_indexes:
            start = index['start']
            end   = index['end']

            res = prediction_result.eval(session=sess, feed_dict={
                x_premise: x_dev_premise[start:end],
                x_hypothesis: x_dev_hypothesis[start:end],
                embedding_arr: pretrained_weights,
                y: y_dev[start:end],
                keep_prob: 1.0,
                keep_prob_lstm: 1.0,
                is_training: False
            })
            dev_pre_result += np.sum(res)
        print(dev_pre_result)
        acc = dev_pre_result/dev_size

        print('epoch: ', epoch, '  train_acc:', train_acc, '  dev_acc:' , acc)

        logger.log_scalar('train_acc', train_acc, epoch)
        logger.log_scalar('dev_acc', acc, epoch)
        logger.log_histogram('w1', hist_sum_dic['rw1_sum'] / train_size, epoch)
        logger.log_histogram('w2', hist_sum_dic['rw2_sum'] / train_size, epoch)
        logger.log_histogram('w3', hist_sum_dic['rw3_sum'] / train_size, epoch)
        logger.log_histogram('h1', hist_sum_dic['rh1_sum'] / train_size, epoch)
        logger.log_histogram('h2', hist_sum_dic['rh2_sum'] / train_size, epoch)
        logger.log_histogram('h3', hist_sum_dic['rh3_sum'] / train_size, epoch)
        logger.log_histogram('v' , hist_sum_dic['rv_sum']  / train_size, epoch)
        logger.log_histogram('b' , hist_sum_dic['rb_sum']  / train_size, epoch)
        # Early_Stopping =========================
        # if es.validate(acc):
        #     break


    # testデータでの評価 ========================================================
    test_pre_result = 0
    for index in test_batch_indexes:
        start = index['start']
        end   = index['end']

        res = prediction_result.eval(session=sess, feed_dict={
            x_premise: x_test_premise[start:end],
            x_hypothesis: x_test_hypothesis[start:end],
            embedding_arr: pretrained_weights,
            y: y_test[start:end],
            keep_prob: 1.0,
            keep_prob_lstm: 1.0,
            is_training: False
        })
        test_pre_result += np.sum(res)
    acc = test_pre_result / test_size

    print('epoch: ', epoch, '  test_acc: ', acc)
