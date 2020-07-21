import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Iterable, Tuple, Dict
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import math
from easytune.TuneKeras import TuneKeras

batch_size = 64
train_batches = 1024
test_batches = 10


def sin_generator(start: int, num: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for i in range(start, start + num, batch_size):
        features = np.zeros((batch_size, 32), dtype=np.float32)
        labels = np.zeros((batch_size, 1), dtype=np.float32)
        for b in range(0, batch_size, 1):
            features[b, :] = i + b
            labels[b, :] = math.sin(i + b)

        yield features,labels


def train_generator() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    return sin_generator(-train_batches // 2, train_batches)


def test_generator() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    return sin_generator(-test_batches // 2, test_batches)


def build_model(lr: float, loss: str, act_in: str, act_out: str) -> tf.keras.Model:
    input = k.Input(shape=(32, ), dtype=tf.float32)
    x = k.layers.Dense(units=128, activation=act_in)(input)
    x = k.layers.Dense(units=32, activation=act_in)(x)
    output = k.layers.Dense(units=1, activation=act_out, name='out')(x)

    model = k.Model(inputs=[input,],  outputs=[output, ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss={'out': loss})
    return model


params = {'lr': [0.001, 0.0005],
          'loss': ['mean_squared_error'],
          'act_in': ['relu', 'swish', 'tanh'],
          'act_out': ['tanh', 'sigmoid']}

kt = TuneKeras(params, train_generator, 1000, test_generator=test_generator, test_batches=test_batches)

model = kt.search(build_model, epochs=3)
model.save("./sin.h5")