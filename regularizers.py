from keras import backend as K
from abc import ABC, abstractmethod
import utils

# TODO: create regularization for each conv kernel separately?


class NormReg(ABC):
    def __init__(self, coeff):
        self._coeff = coeff
        self._original_weights = None

    @abstractmethod
    def _norm(self, weight_matrix):
        pass

    def __call__(self, weight_matrix):
        if self._original_weights is None and weight_matrix._uses_learning_phase:
            self._original_weights = weight_matrix
        elif not weight_matrix._uses_learning_phase:
            return 0  # this is an ugly fix
        diff_matrix = weight_matrix - self._original_weights
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