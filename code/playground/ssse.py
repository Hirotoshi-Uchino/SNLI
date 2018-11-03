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
    nlp_path       = '/media/hirotoshi/BigData/ml/NLP/'
    snli_data_path = nlp_path + '/SNLI/snli_1.0/'
    embd_data_path = nlp_path +'/word_embed/'

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
    classes    = 3

    epochs = 20

    pretrained_weights = np.load(embd_data_npy)

    x_train_premise    = snli.x_train[0]
    x_train_hypothesis = snli.x_train[1]
    y_train            = snli.y_train

    train_size = x_train_premise.shape[0]
    n_batches  = train_size // batch_size

    x_dev_premise    = snli.x_dev[0]
    x_dev_hypothesis = snli.x_dev[1]
    y_dev            = snli.y_dev

    dev_size      = x_dev_premise.shape[0]
    n_batches_dev = dev_size // batch_size


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
    with tf.name_scope("lstm"):
        lstm_units = 200
        with tf.variable_scope('lstm_01', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
            lstm_out_pre, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=x_pre_embed, dtype=tf.float32)
            lstm_out_hyp, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=x_hyp_embed, dtype=tf.float32)

            out_pre_01 = tf.concat(lstm_out_pre, 2)
            out_hyp_01 = tf.concat(lstm_out_hyp, 2)

    # make inputs for the next lstm layer
    # with tf.name_scope("make_input_01"):
    #     pre_input = tf.concat([x_pre_embed, out_pre_01], axis=2)
    #     hyp_input = tf.concat([x_hyp_embed, out_hyp_01], axis=2)
    #
    #     pre_input = batch_norm_wrapper(pre_input, is_training)
    #     hyp_input = batch_norm_wrapper(hyp_input, is_training)
    #
    # # lstm
    # with tf.name_scope("lstm2"):
    #     lstm_units = 300
    #     with tf.variable_scope('lstm_02', reuse=tf.AUTO_REUSE):
    #         cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
    #         lstm_out_pre, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=pre_input, dtype=tf.float32)
    #         lstm_out_hyp, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=hyp_input, dtype=tf.float32)
    #
    #         out_pre_last_layer = tf.concat(lstm_out_pre, 2)
    #         out_hyp_last_layer = tf.concat(lstm_out_hyp, 2)

    out_pre_last_layer = out_pre_01
    out_hyp_last_layer = out_hyp_01

    # 全結合層に入れる準備
    v_pre = tf.reduce_max(out_hyp_last_layer, 1)
    v_hyp = tf.reduce_max(out_hyp_last_layer, 1)

    v_input = tf.concat([v_pre, v_hyp, tf.abs(tf.subtract(v_pre, v_hyp)),
                         tf.multiply(v_pre, v_hyp)], axis=-1)

    # 全結合nn層
    with tf.name_scope("nn1"):
        nn_units = 300
        w1  = tf.Variable(tf.truncated_normal(([lstm_units*8, nn_units]), stddev=0.1), name='w1')
        h1  = tf.Variable(tf.zeros([nn_units]), name='h1')
        alpha1 = tf.Variable(tf.zeros([nn_units]))
        out = tf.matmul(v_input, w1) + h1
        out = batch_norm_wrapper(out, is_training)
        out = prelu(out, alpha1)
        out = tf.nn.dropout(out, keep_prob)
        # out = tf.nn.relu(out)
        # out = tf.nn.softmax(tf.matmul(v_input, w1) + h1)


    with tf.name_scope("nn2"):
        w2 = tf.Variable(tf.truncated_normal(([nn_units, nn_units]), stddev=0.1))
        h2 = tf.Variable(tf.zeros([nn_units]))
        alpha2 = tf.Variable(tf.zeros([nn_units]))
        out = tf.matmul(out, w2) + h2
        out = batch_norm_wrapper(out, is_training)
        out = prelu(out, alpha2)
        out = tf.nn.dropout(out, keep_prob)
        # out = tf.nn.relu(out)
        # out = tf.nn.softmax(tf.matmul(v_input, w1) + h1)

    # 出力層
    with tf.name_scope("inference"):
        v = tf.Variable(tf.truncated_normal(([nn_units, classes]), stddev=0.1))
        b = tf.Variable(tf.zeros([classes]))

        out = tf.nn.softmax(tf.matmul(out, v) + b)

    # # 誤差
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(out * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                                                  reduction_indices=[1]))

    # # 訓練
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    train_step = optimizer.minimize(cross_entropy)

    # # 評価
    correct_prediction = tf.equal(tf.argmax(y, -1), tf.arg_max(out, -1))
    prediction_result  = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(prediction_result)

    # sessionの準備
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # sess = tf_debug.localclidebugwrappersession(sess)

    summary_writer = tf.summary.FileWriter("../../logs/playground/ssse", sess.graph)

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
             }))
            sess.run(train_step, feed_dict={
                x_premise: x_train_premise[start:end],
                x_hypothesis: x_train_hypothesis[start:end],
                embedding_arr: pretrained_weights,
                y: y_train[start:end],
                keep_prob: 1.0,
                keep_prob_lstm: 1.0,
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
                keep_prob_lstm: 1.0
            })

            dev_pre_results.append(res)
        print('epoch: ', epoch,
              # 'dev_pre_results', dev_pre_results,
              'val_acc', np.mean(dev_pre_results))

