import pickle
import constants
from visualization import Visualizer
import numpy as np

with open(constants.TRACKERS_DIRECTORY + 'tracking_baseline_14p4.pickle', 'rb') as pickle_in:
    tracker = pickle.load(pickle_in)

visualizer = Visualizer(tracker)
jump = 1
visualizer.plot_distances_from_initialization(jump=jump)
visualizer.plot_generalization_error(jump=jump)