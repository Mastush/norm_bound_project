import constants
import pickle
import tracker
import utils
import os
import numpy as np


class Comparator:

    def __init__(self, trackers):
        self._trackers = trackers
        self._best_accuracies = None

    def _create_best_accuracies(self):
        best_accuracies = []
        for tracker in self._trackers:
            accuracy_per_epoch = tracker.get_val_accuracy_per_epoch()
            best_epoch = np.argmax(accuracy_per_epoch)
            best_accuracy = accuracy_per_epoch[best_epoch]
            best_accuracies.append((tracker.get_name(), best_accuracy, best_epoch))
        return best_accuracies

    def get_best_accuracies(self):
        if self._best_accuracies is None:
            self._best_accuracies = self._create_best_accuracies()
        return self._best_accuracies

    def get_normalized_distances_at_best_epochs(self):
        best_accuracies = self.get_best_accuracies()
        best_epochs = [acc_tuple[-1] for acc_tuple in best_accuracies]
        normalized_distances_at_best_epochs = []
        for i in range(len(self._trackers)):
            tracker = self._trackers[i]
            distances = np.asarray(tracker.get_distance_from_original_weights_per_epoch())
            distances = distances.T

            initial_weights = np.asarray(tracker.get_original_weights())
            initial_weights_norms = [utils.squared_frobenius_norm(initial_weights[i]) for i in range(len(initial_weights))]
            normalized_distances = (distances / np.expand_dims(initial_weights_norms, axis=-1)).T
            normalized_distances_at_best_epochs.append((tracker.get_name(),
                                             normalized_distances[best_epochs[i]]))
        return normalized_distances_at_best_epochs

    def get_generalization_error_at_best_epochs(self):
        best_accuracies = self.get_best_accuracies()
        best_epochs = [acc_tuple[-1] for acc_tuple in best_accuracies]
        gen_errors_at_best_epochs = []
        for i in range(len(self._trackers)):
            tracker = self._trackers[i]
            gen_errors_at_best_epochs.append((tracker.get_name(),
                                             tracker.get_0_1_generalization_error_per_epoch()[best_epochs[i]]))
        return gen_errors_at_best_epochs


trackers = []
for tracker_name in os.listdir(constants.TRACKERS_DIRECTORY):
    with open(constants.TRACKERS_DIRECTORY + tracker_name, 'rb') as pickle_in:
        trackers.append(pickle.load(pickle_in))

comparator = Comparator(trackers)
best_epochs = comparator.get_best_accuracies()
normalized_distances = comparator.get_normalized_distances_at_best_epochs()
errors = comparator.get_generalization_error_at_best_epochs()
a = 5
