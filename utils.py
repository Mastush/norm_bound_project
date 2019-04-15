import numpy as np
import keras.backend as K


def squared_frobenius_norm(weight_matrix):
    return np.sum(np.square(weight_matrix))


def keras_squared_frobenius_norm(weight_matrix):
    return K.sum(K.square(weight_matrix))
