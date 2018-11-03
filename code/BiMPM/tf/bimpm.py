import numpy as np

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import tensorflow as tf

from lstm_nn3 import SNLI

if __name__ == '__main__':
    maxlen = 78

    DEBUG = True
    # DEBUG = False

    NLP_path = '/media/hirotoshi/BigData/ml/NLP/'
    embd_data_path = NLP_path + '/word_embed/'
    snli_data_path = NLP_path + '/SNLI/snli_1.0/'

    model_path = './'

    embd_data_npy = NLP_path + '/SNLI/embedding_matrix/snli_glove.6B.300d.npy'

    snli_pickle_path = NLP_path + '/SNLI/pickles/snli.pickle'

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

    # placeholder群
    x_premise = tf.placeholder(tf.int32, [None, maxlen])
    x_hypothesis = tf.placeholder(tf.int32, [None, maxlen])
    y = tf.placeholder(tf.float32, [None, classes])
    embedding_arr = tf.placeholder(tf.float32, [34607, embedding_dim])

    # embedding (Word Representation Layer))
    with tf.name_scope("Word_Representation_Layer"):
        x_pre_embed = tf.nn.embedding_lookup(embedding_arr, x_premise)
        x_hyp_embed = tf.nn.embedding_lookup(embedding_arr, x_hypothesis)

    # Context Representation Layer
    with tf.name_scope("Context_Representation_Layer"):
        lstm_units = 200
        cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, use_peepholes=True)
        lstm_out_pre, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,
                                                          inputs=x_pre_embed, dtype=tf.float32)
        lstm_out_hyp, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,
                                                          inputs=x_hyp_embed, dtype=tf.float32)

        # 変数は、Zhiguo et al.[2017]に準ずる
        h_p_f = lstm_out_pre[0]
        h_p_b = lstm_out_pre[1]
        h_q_f = lstm_out_hyp[0]
        h_q_b = lstm_out_hyp[1]

    with tf.name_scope("Matching_Layer"):
        perspectives = 10         # => this number corresponds "l" in the paper
        # Full-Matching
        W1 = tf.Variable(tf.truncated_normal(([perspectives, lstm_units]), stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal(([perspectives, lstm_units]), stddev=0.1))

        h_q_f_last  = h_q_f[:, -1, :]
        h_q_b_first = h_q_f[:,  0, :]

        # m_p = tf.Variable(tf.zeros([maxlen, perspectives]), trainable=False)
        # m_b = tf.Variable(tf.zeros([maxlen, perspectives]), trainable=False)

        m_p = []
        m_b = []

        for i in range(maxlen):
            m_p.append(tf.matmul(W1*h_p_f[:, i, :], W1*h_q_f_last, transpose_b=True))
            m_b.append(tf.matmul(W2*h_q_b[:, i, :], W2*h_q_b_first, transpose_b=True))



    with tf.name_scope("Aggregation_Layer"):

        pass


    with tf.name_scope("Prediction_Layer"):
        pass

    # last_out_pre = lstm_out_pre[:, -1, :]
    # last_out_hyp = lstm_out_hyp[:, -1, :]
    # lstm_last_out = tf.concat([last_out_pre, last_out_hyp], axis=1)
    # lstm_last_out = tf.nn.dropout(lstm_last_out, keep_prob_lstm)

    # V = tf.Variable(tf.truncated_normal(([nn_units, classes]), stddev=0.1))
    # b = tf.Variable(tf.zeros([classes]))
    #
    # out = tf.nn.softmax(tf.matmul(out2, V) + b)
    #
    # # 誤差
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(out * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
    #                                               reduction_indices=[1]))
    #
    # # 訓練
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    # train_step = optimizer.minimize(cross_entropy)
    #
    # # 評価
    # correct_prediction = tf.equal(tf.argmax(y, -1), tf.arg_max(out, -1))
    # prediction_result = tf.cast(correct_prediction, tf.float32)
    # accuracy = tf.reduce_mean(prediction_result)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(epochs):

        for i in range(n_batches):
            if i % 10 == 0:
                print("batch: {}/{}".format(i, n_batches))
            start = i * batch_size
            end = start + batch_size

            print(sess.run(m_p, feed_dict={
                x_premise: x_train_premise[start:end],
                x_hypothesis: x_train_hypothesis[start:end],
                embedding_arr: pretrained_weights,
                y: y_train[start:end],
            }))
            print(sess.run(m_b, feed_dict={
                x_premise: x_train_premise[start:end],
                x_hypothesis: x_train_hypothesis[start:end],
                embedding_arr: pretrained_weights,
                y: y_train[start:end],
            }))
        #     sess.run(train_step, feed_dict={
        #         x_premise: x_train_premise[start:end],
        #         x_hypothesis: x_train_hypothesis[start:end],
        #         embedding_arr: pretrained_weights,
        #         y: y_train[start:end],
        #         keep_prob: 1.0,
        #         keep_prob_lstm: 1.0,
        #         is_training: True
        #     })
        #
        # print("evaluating by using dev data....")
        # dev_pre_results = []
        # for i in range(n_batches_dev):
        #     res = prediction_result.eval(session=sess, feed_dict={
        #         x_premise: x_dev_premise,
        #         x_hypothesis: x_dev_hypothesis,
        #         embedding_arr: pretrained_weights,
        #         y: y_dev,
        #         keep_prob: 1.0,
        #         keep_prob_lstm: 1.0,
        #         is_training: False
        #     })
        #     dev_pre_results.append(res)

        # print('epoch: ', epoch,
        #       # 'dev_pre_results', dev_pre_results,
        #       'val_acc', np.mean(dev_pre_results))

