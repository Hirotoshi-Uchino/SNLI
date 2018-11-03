import pandas as pd
import numpy as np

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from models_DA_ESIM import decomposable_attention, esim, esim_qrnn
from nlp_util import create_embedding_matrix

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import keras.backend as K
# from tensorflow.python import debug as tf_debug

class SNLI:

    def __init__(self, train_df, dev_df, test_df):
        self.x_train = None
        self.x_dev   = None
        self.x_test  = None
        self.y_train = None
        self.y_dev   = None
        self.y_test  = None
        self.tokenizer = None
        self.preprocessing(train_df, dev_df, test_df)


    def preprocessing(self, train_df, dev_df, test_df):
        train_df = train_df[train_df["gold_label"] != "-"].fillna("")
        dev_df = dev_df[dev_df["gold_label"] != "-"].fillna("")
        test_df = test_df[test_df["gold_label"] != "-"].fillna("")

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df["sentence1"])
        tokenizer.fit_on_texts(train_df["sentence2"])
        tokenizer.fit_on_texts(dev_df["sentence1"])
        tokenizer.fit_on_texts(dev_df["sentence2"])
        tokenizer.fit_on_texts(test_df["sentence1"])
        tokenizer.fit_on_texts(test_df["sentence2"])

        seq_train1 = tokenizer.texts_to_sequences(train_df["sentence1"])
        seq_train2 = tokenizer.texts_to_sequences(train_df["sentence2"])
        seq_dev1   = tokenizer.texts_to_sequences(dev_df["sentence1"])
        seq_dev2   = tokenizer.texts_to_sequences(dev_df["sentence2"])
        seq_test1  = tokenizer.texts_to_sequences(test_df["sentence1"])
        seq_test2  = tokenizer.texts_to_sequences(test_df["sentence2"])

        maxlen = 78

        X_train1 = sequence.pad_sequences(seq_train1, maxlen=maxlen)
        X_train2 = sequence.pad_sequences(seq_train2, maxlen=maxlen)
        self.x_train = [X_train1, X_train2]

        y_label = {"contradiction": 0, "entailment": 1, "neutral": 2}
        y_train = [y_label[i] for i in train_df["gold_label"]]
        self.y_train = np_utils.to_categorical(y_train, 3)

        X_dev1 = sequence.pad_sequences(seq_dev1, maxlen=maxlen)
        X_dev2 = sequence.pad_sequences(seq_dev2, maxlen=maxlen)
        self.x_dev = [X_dev1, X_dev2]

        y_dev = [y_label[i] for i in dev_df["gold_label"]]
        self.y_dev = np_utils.to_categorical(y_dev, 3)

        X_test1 = sequence.pad_sequences(seq_test1, maxlen=maxlen)
        X_test2 = sequence.pad_sequences(seq_test2, maxlen=maxlen)
        self.x_test = [X_test1, X_test2]

        y_test = [y_label[i] for i in test_df["gold_label"]]
        self.y_test = np_utils.to_categorical(y_test, 3)

        self.tokenizer = tokenizer


if __name__ == '__main__':

    # DEBUG = True
    DEBUG = False

    model_name = "Decom_Attr"
    # model_name = "ESIM"
    # model_name = "ESIM_QRNN"


    NLP_path       = '/media/hirotoshi/BigData/ml/NLP/'
    snli_data_path = NLP_path + '/SNLI/snli_1.0/'
    embd_data_path = NLP_path +'/word_embed/'

    model_path = './'

    train_df = pd.read_csv(snli_data_path + "/snli_1.0_train.txt", sep="\t", header=0)
    dev_df   = pd.read_csv(snli_data_path + "/snli_1.0_dev.txt", sep="\t", header=0)
    test_df  = pd.read_csv(snli_data_path + "/snli_1.0_test.txt", sep="\t", header=0)


    snli = SNLI(train_df=train_df, dev_df=dev_df, test_df=train_df)

    # ==============================================
    # create embedding_matrix and save (if needed)
    # ==============================================
    # embd_data = embd_data_path + '/glove.6B.300d.txt'
    # create_embedding_matrix(embd_data, snli.tokenizer.word_index, embedding_dim=300,
    #                         output_file_name=NLP_path + '/SNLI/embedding_matrix/snli_glove.6B.300d.npy')


    embd_data_npy = NLP_path + '/SNLI/embedding_matrix/snli_glove.6B.300d.npy'


    # if DEBUG:
    #     sess = K.get_session()
    #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #     K.set_session(sess)


    nb_epochs = 20
    if model_name == "Decom_Attr":
        batch_size = 2000  # For Decom-Attr
        model = decomposable_attention(pretrained_embedding=embd_data_npy,
                                   projection_dim=300, maxlen=78, num_class=3)

    if model_name == "ESIM":
        batch_size = 100  # For ESIM
        if DEBUG:
            batch_size = 1
        model = esim(pretrained_embedding=embd_data_npy,
                     maxlen=78, lstm_dim=300, dense_dim=300, num_class=3)

    if model_name == "ESIM_QRNN":
        batch_size = 1000  # For ESIM_QRNN
        model = esim_qrnn(pretrained_embedding=embd_data_npy,
                     maxlen=78, qrnn_dim=300, dense_dim=300, num_class=3)


    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)


    model.fit(snli.x_train, snli.y_train,
              batch_size=batch_size,
              epochs=nb_epochs,
              validation_data=(snli.x_dev, snli.y_dev),
              # shuffle=True
              shuffle=True, callbacks=[early_stopping]
              )

    # y_pred = model.predict_classes(snli.x_test, batch_size=batch_size).flatten()
    loss, binary_cross_entropy, acc = model.evaluate(snli.x_test, snli.y_test, batch_size=batch_size)
    # score, acc = model.evaluate(snli.x_dev, snli.y_dev, batch_size=batch_size)

    model_filename = model_path + 'model_' + model_name +'.hdf5'
    model.save(model_filename)
    # print()
    print("Test loss:", loss)
    print("Test accuracy:", acc)

