import numpy as np
from tensorflow_core.python import Constant
from tensorflow_core.python.keras import Input, Model

from tensorflow_core.python.keras.layers import LSTM, Embedding, Dropout, Dense, BatchNormalization

from layer import Attention_COSTUM


def get_model(X, N_class, total_words=86627, EMBEDDING_DIM=100, maxlen=53):

    embeddings_index = {}

    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((total_words, EMBEDDING_DIM))
    for word, i in X.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    inp = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(total_words,
                          EMBEDDING_DIM,
                          embeddings_initializer=Constant(embedding_matrix),
                          input_length=maxlen,
                          trainable=False)(inp)
    x = LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(embedding)
    x = Dropout(0.25)(x)
    merged = Attention_COSTUM(maxlen)(x)
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.25)(merged)
    merged = BatchNormalization()(merged)
    outp = Dense(N_class, activation='softmax')(merged)

    AttentionLSTM = Model(inputs=inp, outputs=outp)
    AttentionLSTM.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return AttentionLSTM