import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K

from keras import backend as K

from layers import ElementwiseProduct

MAX_LEN = 30

def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding

def full_matching(input1, input2, multi_perspective):
    pass



def BiMPM(pretrained_embedding='../data/fasttext_matrix.npy',
          lstm_dim=300, dense_dim=300, dense_dropout=0.2,
          multi_perspective=1,
          lr=1e-3, activation='relu', maxlen=MAX_LEN, num_class=1):
    # Based on arXiv:1609.06038
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    # Full-Matching ====>>>>

    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))


    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True), merge_mode=None)
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Multi-perspective Matching
    ewp = ElementwiseProduct(multi_perspective)
    q1_ewp = ewp(q1_encoded)
    q2_ewp = ewp(q2_encoded)

    # cosine...
    m = Dot(axes=2)

    #


    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])
    return model