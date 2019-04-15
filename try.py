import pickle
import constants

with open(constants.TRACKERS_DIRECTORY + 'tracking_baseline.pickle', 'rb') as pickle_in:
    b = pickle.load(pickle_in)

a = 5
