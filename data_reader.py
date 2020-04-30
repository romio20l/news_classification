import json

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


class DataInput:
    def __init__(self, filename):
        self.headlines = []
        self.category = []
        with open(filename) as f:
            for line in f.readlines():
                j = json.loads(line)
                self.headlines.append(j['headline'])
                self.category.append(j['category'])
            f.close()

    def get_train_test_data(self, split_factor=0.8):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.headlines)
        train_sequences = tokenizer.texts_to_sequences(self.headlines)
        self.word_list = tokenizer.word_index
        train_padded = pad_sequences(train_sequences, padding='post')

        label_tokenizer = Tokenizer()
        label_tokenizer.fit_on_texts(self.category)
        labels = [label[0] for label in label_tokenizer.texts_to_sequences(self.category)]
        self.label_len = len(label_tokenizer.word_index) + 1
        training_label_seq = np.array(labels)

        split_index = int(len(train_padded) * split_factor)
        X_train = train_padded[0:split_index]
        Y_train = training_label_seq[0:split_index]
        X_test = train_padded[split_index:]
        Y_test = training_label_seq[split_index:]

        return X_train, Y_train, X_test, Y_test
