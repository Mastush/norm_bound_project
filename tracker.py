import utils
import numpy as np


class Tracker:  # TODO: DEBUG!!!!!!!!

    def __init__(self, name):
        self._name = name
        self._original_weights = None
        self._current_weights = None
        self._distance_from_original_weights_per_epoch = []
        self._0_1_generalization_error_per_epoch = []
        self._loss_generalization_error_per_epoch = []

    def update_original_layer_weights(self, layer_list):
        self._original_weights = []
        for layer in layer_list:
            if len(layer.get_weights()) > 0:
                self._original_weights.append(layer.get_weights()[0])

    def _update_distance_from_original_weights(self, layer_list):
        current_weights = []
        for layer in layer_list:
            if len(layer.get_weights()) > 0:
                current_weights.append(layer.get_weights()[0])

        distance_from_original_weights = []
        for i in range(len(self._original_weights)):
            diff_matrix = current_weights[i] - self._original_weights[i]
            distance_from_original_weights.append(utils.squared_frobenius_norm(diff_matrix))
        self._distance_from_original_weights_per_epoch.append(distance_from_original_weights)

    def _update_generalization_errors(self, logs):
        self._0_1_generalization_error_per_epoch.append(abs(logs['categorical_accuracy'] -
                                                            logs['val_categorical_accuracy']))
        self._loss_generalization_error_per_epoch.append(abs(logs['loss'] - logs['val_loss']))

    def update_tracker(self, model, logs):
        self._update_distance_from_original_weights(model.layers)
        self._update_generalization_errors(logs)

    def get_name(self): return self._name