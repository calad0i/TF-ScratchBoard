import tensorflow as tf
import numpy as np
from tensorflow import keras

X = np.zeros(1000, dtype=float)
Y = np.zeros(1000, dtype=float)
for i in range(1000):
    X[i] = i
    Y[i] = 2 * i + (np.random.random_sample() - 0.5) * 5 + 30
MX = np.vstack([X, np.ones(len(X))]).T
aa, bb = np.linalg.lstsq(MX, Y, rcond=None)[0]

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)

variables = [a, b]

x = tf.constant(X, dtype=tf.float32)
y = tf.constant(Y, dtype=tf.float32)

num_epoch = 10000
num_epoch2 = 100

optimizer = tf.keras.optimizers.SGD(learning_rate=2e-10)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * x + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print((aa, bb), (a.numpy(), b.numpy()))
keras.preprocessing.sequence.