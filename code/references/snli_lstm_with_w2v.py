"""Trains a LSTM with Word2Vec on the SNLI dataset.

https://nlp.stanford.edu/projects/snli/

Get to 80.12% test accuracy after 18 epochs. 
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import importlib
import sys
#importlib.reload(sys)
#sys.setdefaultencoding('utf-8')

EMBEDDING_FILE  = '/home/hirotoshi/projects/ml/mathG/SNLI/data/GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = "/home/hirotoshi/projects/ml/mathG/SNLI/data/snli_1.0/snli_1.0_train.txt"
TEST_DATA_FILE  = "/home/hirotoshi/projects/ml/mathG/SNLI/data/snli_1.0/snli_1.0_test.txt"
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

def text_to_tokens(text):
    return text.lower()

def get_label_index_mapping():
    return {"neutral": 0, "contradiction": 1, "entailment": 2, "-": 3}
    
def create_embedding_matrix(word_index):
    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    return embedding_matrix

def load_data():
    train_df = pd.read_csv(TRAIN_DATA_FILE, sep="\t", usecols=["sentence1", "sentence2", "gold_label"], dtype={"sentence1": str, "sentence2": str, "gold_label": str})
    train_df.fillna("", inplace=True)
    sentence1 = train_df["sentence1"].apply(text_to_tokens) # すべて小文字に
    sentence2 = train_df["sentence2"].apply(text_to_tokens)
    y = train_df["gold_label"].map(get_label_index_mapping()) # 回答ラベルを準備
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(sentence1 + sentence2)
    sequences1 = tokenizer.texts_to_sequences(sentence1) # textからvector値へ変換
    sequences2 = tokenizer.texts_to_sequences(sentence2)
    X1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
    X2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)
    perm = np.random.permutation(len(X1)) # shuffleのため
    num_train   = int(len(X1)*(1-VALIDATION_SPLIT))
    train_index = perm[:num_train]
    valid_index = perm[num_train:]
    X1_train    = X1[train_index]
    X2_train    = X2[train_index]
    y_train     = y[train_index]
    X1_valid    = X1[valid_index]
    X2_valid    = X2[valid_index]
    y_valid     = y[valid_index]
    return (X1_train, X2_train, y_train), (X1_valid, X2_valid, y_valid), tokenizer

def StaticEmbedding(embedding_matrix):
    input_dim, output_dim = embedding_matrix.shape
    return Embedding(input_dim,
            output_dim,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)
    
def entail(feat1, feat2, num_dense=300):
    x = concatenate([feat1, feat2])
    x = Dropout(rate_drop_dense)(x)
    x = BatchNormalization()(x) # 前の層の出力を正規化
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = BatchNormalization()(x)
    return x

def build_model(output_dim, embedding_matrix, num_lstm=300):
    sequence1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') # Keras Tensorのインスタンス化
    sequence2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
   
    # Embedding
    embed = StaticEmbedding(embedding_matrix)
    embedded_sequences1 = embed(sequence1_input)
    embedded_sequences2 = embed(sequence2_input)
    
    # Encoding
    encode = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    feat1 = encode(embedded_sequences1)
    feat2 = encode(embedded_sequences2)
   
    x = entail(feat1, feat2)
    preds = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=[sequence1_input, sequence2_input], outputs=preds)
    return model

def run():
    num_class = len(get_label_index_mapping())
    (X1_train, X2_train, y_train), (X1_valid, X2_valid, y_valid), tokenizer = load_data()
    Y_train, Y_valid = to_categorical(y_train, num_class), to_categorical(y_valid, num_class)
    embedding_matrix = create_embedding_matrix(tokenizer.word_index) 
    model = build_model(output_dim=num_class, embedding_matrix=embedding_matrix) 
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    hist = model.fit([X1_train, X2_train], Y_train,
            validation_data=([X1_valid, X2_valid], Y_valid),
            epochs=200, batch_size=2048, shuffle=True,
            callbacks=[early_stopping])

if __name__ == "__main__":
    run()



