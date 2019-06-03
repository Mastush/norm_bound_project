import keras
import pickle
import constants
from visualization import Visualizer
import utils
import os
import numpy as np

# tracker_name = 'baseline_model_reg_None_coeff_0.0_18_april.pickle'
# for tracker_name in os.listdir(constants.TRACKERS_DIRECTORY):
#     with open(constants.TRACKERS_DIRECTORY + tracker_name, 'rb') as pickle_in:
#         tracker = pickle.load(pickle_in)
#     visualizer = Visualizer(tracker, target_directory=constants.RESULTS_DIR + utils.clip_filename(tracker_name))
#     jump = 1
#     visualizer.plot_distances_from_initialization(jump=jump)
#     visualizer.plot_normalized_distances_from_initialization(jump=jump)
#     visualizer.plot_generalization_error(jump=jump)
#     visualizer.plot_accuracy(jump=jump)


input_layer = keras.layers.Input((6, 6, 1))
last_layer = keras.layers.Conv2D(1, (3, 3))(input_layer)
last_layer = keras.layers.Flatten()(last_layer)
output_layer = keras.layers.Dense(2)(last_layer)

model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=constants.OPTIMIZER, loss=constants.LOSS)
a = 5