from keras.regularizers import Regularizer
from keras import backend as K
from abc import ABC, abstractmethod
import utils


class NormReg(ABC, Regularizer):  # TODO: enable serialization!!!
    def __init__(self, coeff):
        self._coeff = coeff

    @abstractmethod
    def _norm(self, weight_matrix):
        pass

    def __call__(self, weight_matrix):
        diff_matrix = weight_matrix - K.eval(weight_matrix)
        return self._coeff * self._norm(diff_matrix)


class FrobReg(NormReg):  # does not differentiate between kernels in convolution layers!
    def __init__(self, coeff):
        super().__init__(coeff)

    def _norm(self, weight_matrix):
        return K.sqrt(K.sum(K.square(weight_matrix)))


class SquaredFrobReg(NormReg):
    def __init__(self, coeff):
        super().__init__(coeff)

    def _norm(self, weight_matrix):
        return utils.keras_squared_frobenius_norm(weight_matrix)


class CustomNormReg(NormReg):
    def __init__(self, coeff, norm_func):
        super().__init__(coeff)
        self._norm_func = norm_func

    def _norm(self, weight_matrix):
        return self._norm_func(weight_matrix)
