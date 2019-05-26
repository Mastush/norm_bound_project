import utils
import pickle
import constants
from keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np


class Tracker:
    def __init__(self, name, x_train, y_train, x_val, y_val, compute_training_error=False,
                 examples_for_training_error_approx=None, save=True):
        self._name = name
        self._original_weights = None
        self._last_weights = None
        self._x_train = x_train
        self._y_train = y_train
        self._x_val = x_val
        self._y_val = y_val
        self._compute_training_error = compute_training_error
        self._examples_for_training_error_approx = examples_for_training_error_approx
        self._save = save
        self._distance_from_original_weights_per_epoch = []
        self._0_1_generalization_error_per_epoch = []
        self._loss_generalization_error_per_epoch = []
        self._training_accuracy_per_epoch = []
        self._training_loss_per_epoch = []
        self._val_accuracy_per_epoch = []
        self._val_loss_per_epoch = []

    def update_original_layer_weights(self, layer_list):
        self._original_weights = []
        for layer in layer_list:
            if len(layer.get_weights()) > 0:
                self._original_weights.append(layer.get_weights()[0])

    def update_last_layer_weights(self, layer_list):
        self._last_weights = []
        for layer in layer_list:
            if len(layer.get_weights()) > 0:
                self._last_weights.append(layer.get_weights()[0])

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

    def _update_errors(self, model, logs):
        if self._compute_training_error:
            x_for_training_loss, y_for_training_loss = self._x_train, self._y_train
            if self._examples_for_training_error_approx is not None:
                pairs = list(zip(x_for_training_loss, y_for_training_loss))  # make pairs out of the two lists
                pairs = random.sample(pairs, self._examples_for_training_error_approx)  # pick random pairs
                x_for_training_loss, y_for_training_loss = zip(*pairs)
                x_for_training_loss, y_for_training_loss = np.asarray(x_for_training_loss), \
                                                           np.asarray(y_for_training_loss)

            training_loss, training_accuracy = model.evaluate(x=x_for_training_loss, y=y_for_training_loss,
                                                              batch_size=constants.BATCH_SIZE, verbose=0)
        else:
            training_accuracy = logs['categorical_accuracy']
            training_loss = logs['loss']

        val_loss, val_accuracy = model.evaluate(x=self._x_val, y=self._y_val,
                                                batch_size=constants.BATCH_SIZE, verbose=0)
        self._val_accuracy_per_epoch.append(val_accuracy)
        self._val_loss_per_epoch.append(val_loss)
        self._training_accuracy_per_epoch.append(training_accuracy)
        self._training_loss_per_epoch.append(training_loss)

        self._0_1_generalization_error_per_epoch.append(abs(training_accuracy -
                                                            val_accuracy))
        self._loss_generalization_error_per_epoch.append(abs(training_loss - val_loss))

    def update_tracker(self, model, logs):
        self._update_distance_from_original_weights(model.layers)
        self._update_errors(model, logs)

    def get_name(self): return self._name

    def get_original_weights(self): return self._original_weights

    def get_distance_from_original_weights_per_epoch(self): return self._distance_from_original_weights_per_epoch

    def get_0_1_generalization_error_per_epoch(self): return self._0_1_generalization_error_per_epoch

    def get_loss_generalization_error_per_epoch(self): return self._loss_generalization_error_per_epoch

    def get_val_accuracy_per_epoch(self): return self._val_accuracy_per_epoch

    def get_val_loss_per_epoch(self): return self._val_loss_per_epoch

    def save_tracker(self):
        if self._save:
            self._x_train = None  # no use in saving these...
            self._y_train = None
            self._x_val = None
            self._y_val = None
            with open(constants.TRACKERS_DIRECTORY + "{}.pickle".format(self.get_name()), 'wb') as pickle_out:
                pickle.dump(self, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)


class TrackerForGenerators(Tracker):

    def __init__(self, name, x_train, y_train, x_val, y_val, compute_training_error=False, save=True):
        super().__init__(name, None, None, None, None, compute_training_error=compute_training_error,
                         examples_for_training_error_approx=None, save=save)
        self._x_train = x_train
        self._y_train = y_train
        self._x_val = x_val
        self._y_val = y_val

    def _update_errors(self, model, logs):
        if self._compute_training_error:
            x_for_training_loss, y_for_training_loss = next(self._x_train), next(self._y_train)
            training_loss, training_accuracy = model.evaluate(x=x_for_training_loss, y=y_for_training_loss,
                                                              batch_size=constants.BATCH_SIZE, verbose=0)
        else:
            training_accuracy = logs['categorical_accuracy']
            training_loss = logs['loss']

        val_loss, val_accuracy = model.evaluate(x=self._x_val, y=self._y_val,
                                                batch_size=constants.BATCH_SIZE, verbose=0)
        self._val_accuracy_per_epoch.append(val_accuracy)
        self._val_loss_per_epoch.append(val_loss)
        self._training_accuracy_per_epoch.append(training_accuracy)
        self._training_loss_per_epoch.append(training_loss)

        self._0_1_generalization_error_per_epoch.append(abs(training_accuracy -
                                                            val_accuracy))
        self._loss_generalization_error_per_epoch.append(abs(training_loss - val_loss))

