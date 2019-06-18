import utils
import pickle
import constants
from keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np


class Tracker:
    def __init__(self, name, x_train, y_train, x_val, y_val, compute_training_error=False,
                 examples_for_training_error_approx=None, save=True, top_k_acc=False):
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
        self._singular_values_per_epoch = []
        self._0_1_generalization_error_per_epoch = []
        self._loss_generalization_error_per_epoch = []
        self._training_accuracy_per_epoch = []
        self._training_loss_per_epoch = []
        self._val_accuracy_per_epoch = []
        self._val_loss_per_epoch = []
        self._top_k_acc = top_k_acc
        self._training_top_k_acc_per_epoch = []
        self._val_top_k_acc_per_epoch = []
        self._top_k_generalization_error_per_epoch = []

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

    def _get_training_stats(self, model, logs):
        training_top_k = None

        if self._compute_training_error:
            x_for_training_loss, y_for_training_loss = self._x_train, self._y_train
            if self._examples_for_training_error_approx is not None:
                pairs = list(zip(x_for_training_loss, y_for_training_loss))  # make pairs out of the two lists
                pairs = random.sample(pairs, self._examples_for_training_error_approx)  # pick random pairs
                x_for_training_loss, y_for_training_loss = zip(*pairs)
                x_for_training_loss, y_for_training_loss = np.asarray(x_for_training_loss), \
                                                           np.asarray(y_for_training_loss)
            if self._top_k_acc:
                training_loss, training_accuracy, training_top_k = model.evaluate(x=x_for_training_loss,
                                                                                  y=y_for_training_loss,
                                                                                  batch_size=constants.BATCH_SIZE,
                                                                                  verbose=0)
            else:
                training_loss, training_accuracy = model.evaluate(x=x_for_training_loss, y=y_for_training_loss,
                                                                  batch_size=constants.BATCH_SIZE, verbose=0)
        else:
            if self._top_k_acc: training_top_k = logs['top_k_categorical_accuracy']
            training_accuracy = logs['categorical_accuracy']
            training_loss = logs['loss']

        return training_loss, training_accuracy, training_top_k

    def _update_errors(self, model, logs):
        training_loss, training_accuracy, training_top_k  = self._get_training_stats(model, logs)

        val_loss, val_accuracy = logs['val_loss'], logs['val_categorical_accuracy']
        if self._top_k_acc:
            self._training_top_k_acc_per_epoch.append(training_top_k)
            val_top_k = logs['val_top_k_categorical_accuracy']
            self._val_top_k_acc_per_epoch.append(val_top_k)
            self._top_k_generalization_error_per_epoch.append(abs(training_top_k - val_top_k))
        self._val_accuracy_per_epoch.append(val_accuracy)
        self._val_loss_per_epoch.append(val_loss)
        self._training_accuracy_per_epoch.append(training_accuracy)
        self._training_loss_per_epoch.append(training_loss)

        self._0_1_generalization_error_per_epoch.append(abs(training_accuracy -
                                                            val_accuracy))
        self._loss_generalization_error_per_epoch.append(abs(training_loss - val_loss))

    def _update_singular_values(self, layer_list):
        epoch_singular_values = []
        for layer in layer_list:
            if len(layer.get_weights()) > 0:
                weights = layer.get_weights()[0]
                epoch_singular_values.append(utils.get_weights_spectrum(weights))
        self._singular_values_per_epoch.append(epoch_singular_values)

    def update_tracker(self, model, logs):
        self._update_distance_from_original_weights(model.layers)
        self._update_errors(model, logs)
        self._update_singular_values(model.layers)
        self.save_tracker()

    def get_name(self): return self._name

    def get_original_weights(self): return self._original_weights

    def get_distance_from_original_weights_per_epoch(self): return self._distance_from_original_weights_per_epoch

    def get_0_1_generalization_error_per_epoch(self): return self._0_1_generalization_error_per_epoch

    def get_loss_generalization_error_per_epoch(self): return self._loss_generalization_error_per_epoch

    def get_top_k_acc_generalization_error_per_epoch(self): return self._top_k_generalization_error_per_epoch

    def get_val_accuracy_per_epoch(self): return self._val_accuracy_per_epoch

    def get_val_loss_per_epoch(self): return self._val_loss_per_epoch

    def get_val_top_k_per_epoch(self): return self._val_top_k_acc_per_epoch

    def save_tracker(self):
        if self._save:
            temp_x_train = self._x_train
            temp_y_train = self._y_train
            temp_x_val = self._x_val
            temp_y_val = self._y_val

            self._x_train = None  # no sense in saving these...
            self._y_train = None
            self._x_val = None
            self._y_val = None
            with open(constants.TRACKERS_DIRECTORY + "{}.pickle".format(self.get_name()), 'wb') as pickle_out:
                pickle.dump(self, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

            self._x_train = temp_x_train
            self._y_train = temp_y_train
            self._x_val = temp_x_val
            self._y_val = temp_y_val


class TrackerForTFRecords(Tracker):
    def __init__(self, name, compute_training_error=False, examples_for_training_error_approx=None, save=True,
                 top_k_acc=False):
        """Using this class assumes that the model knows where training data comes from"""
        super().__init__(name, None, None, None, None, compute_training_error=compute_training_error,
                         examples_for_training_error_approx=examples_for_training_error_approx, save=save,
                         top_k_acc=top_k_acc)

    def _get_training_stats(self, model, logs):
        training_top_k = None
        if self._compute_training_error:
            if self._top_k_acc:
                training_loss, training_accuracy, training_top_k = \
                    model.evaluate(verbose=0,
                                   steps=self._examples_for_training_error_approx // constants.BATCH_SIZE)
            else:
                training_loss, training_accuracy = \
                    model.evaluate(verbose=0,
                                   steps=self._examples_for_training_error_approx //constants.BATCH_SIZE)
        else:
            if self._top_k_acc: training_top_k = logs['top_k_categorical_accuracy']
            training_accuracy = logs['categorical_accuracy']
            training_loss = logs['loss']
        return training_loss, training_accuracy, training_top_k
