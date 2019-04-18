import numpy as np
from matplotlib import pyplot as plt
import os


class Visualizer:

    def __init__(self, tracker, target_directory=""):
        self._tracker = tracker
        self._directory = target_directory if (target_directory == "" or target_directory[-1] == '/') \
            else target_directory + '/'
        self._num_of_epochs = len(self._tracker.get_0_1_generalization_error_per_epoch())
        if self._directory != "" and not os.path.exists(self._directory[:-1]):
            os.makedirs(self._directory[:-1])

    def _plot_information_by_epoch(self, values_to_plot, tags, epoch_range, plot_title, filename, ylabel=None):
        assert len(values_to_plot) == len(tags)
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')

        for i in range(len(values_to_plot)):
            plt.plot(epoch_range, values_to_plot[i], color=palette(i), linewidth=1, alpha=0.9, label=tags[i])

        plt.legend(loc=2, ncol=2)

        plt.title(plot_title, loc='left', fontsize=12, fontweight=0, color='black')
        plt.xlabel("num of epochs")
        if ylabel is not None:
            plt.ylabel = ylabel

        plt.savefig(self._directory + filename)

    def plot_distances_from_initialization(self, jump=1):
        distances = np.asarray(self._tracker.get_distance_from_original_weights_per_epoch())
        distances = distances.T
        distances = distances[:, ::jump]
        epoch_range = np.arange(self._num_of_epochs) + 1
        epoch_range = epoch_range[::jump]

        tags = ["Layer {}".format(i + 1) for i in range(len(distances))]

        self._plot_information_by_epoch(distances, tags, epoch_range,
                                        "Distances From Initializations",
                                        "Distances_From_Initializations.png")

    def plot_generalization_error(self, jump=1):
        generalization_error = np.expand_dims(self._tracker.get_0_1_generalization_error_per_epoch(), axis=0)
        generalization_error = generalization_error[:, ::jump]
        epoch_range = np.arange(self._num_of_epochs) + 1
        epoch_range = epoch_range[::jump]

        tags = ["0-1 Generalization Error"]

        self._plot_information_by_epoch(generalization_error, tags, epoch_range, "Generalization Error",
                                        "Generalization_Error")


