import numpy as np
import tensorflow.keras.backend as K


def squared_frobenius_norm(weight_matrix):
    return np.sum(np.square(weight_matrix))


def keras_squared_frobenius_norm(weight_matrix):
    return K.sum(K.square(weight_matrix))


def clip_filename(filename):
    return filename[:filename.rfind('.')]


def get_weights_spectrum(weights):
    if len(weights.shape) > 2:
        rows, cols = 1, weights.shape[-1]
        for i in range(len(weights.shape) - 1):
            rows *= weights.shape[i]
        weights = weights.reshape((rows, cols))
    values = np.linalg.svd(weights, compute_uv=False)
    return values