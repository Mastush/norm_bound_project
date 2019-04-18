import pickle
import constants
from visualization import Visualizer
import utils
from regularizers import SquaredFrobReg

tracker_name = 'baseline_model_with_reg_coeff_1_17_april.pickle'
with open(constants.TRACKERS_DIRECTORY + tracker_name, 'rb') as pickle_in:
    tracker = pickle.load(pickle_in)
visualizer = Visualizer(tracker, target_directory=utils.clip_filename(tracker_name))
jump = 1
visualizer.plot_distances_from_initialization(jump=jump)
visualizer.plot_generalization_error(jump=jump)