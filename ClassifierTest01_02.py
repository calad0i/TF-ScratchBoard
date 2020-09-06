import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt


class LRReducer(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.998
        #print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
        self.model.optimizer.lr.assign(new_lr)


imdb = ks.datasets.imdb

(x2_train, y2_train), (x2_test, y2_test) = imdb.load_data(num_words=100000)

x2_train = ks.preprocessing.sequence.pad_sequences(x2_train,
                                                   value=0,
                                                   padding="post",
                                                   maxlen=256)
x2_test = ks.preprocessing.sequence.pad_sequences(x2_test,
                                                  value=0,
                                                  padding="post",
                                                  maxlen=256)

index = dict([(a, b + 3) for (a, b) in imdb.get_word_index().items()])
index["<PAD>"] = 0
index["^"] = 1
index["<UNK>"] = 2
index["<UNUSED>"] = 3
index_flip = dict([(b, a) for (a, b) in index.items()])


def decode(code):
    return " ".join(index_flip.get(x, "#") for x in code)


def encode(setence):
    return np.array([index.get(x, 2) for x in setence.split()])


loss_fn2 = tf.keras.losses.BinaryCrossentropy()

model2 = tf.keras.models.Sequential([
    ks.layers.Embedding(100000, 16),
    ks.layers.GlobalAveragePooling1D(),
    ks.layers.Dense(256, activation='relu'),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(1, activation='sigmoid')
])

model2.compile(optimizer='adam',
               loss="binary_crossentropy",
               metrics=['accuracy'])

model2.fit(x2_train,
           y2_train,
           batch_size=5000,
           epochs=50,
           verbose=2,
           validation_data=(x2_test, y2_test),
           callbacks=[LRReducer()])
