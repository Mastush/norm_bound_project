from keras import backend as K


class FrobReg:
    def __init__(self, coeff):
        self._coeff = coeff
        self._original_weights = None

    @staticmethod
    def _frobenius_norm(weight_matrix):
        return K.sqrt(K.sum(K.square(weight_matrix)))

    def __call__(self, weight_matrix):
        if self._original_weights is None:
            self._original_weights = weight_matrix
        diff_matrix = weight_matrix - self._original_weights
        return self._coeff * FrobReg._frobenius_norm(diff_matrix)

