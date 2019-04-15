from keras.callbacks import Callback
import pickle
import constants


class TrackerCallback(Callback):

    def __init__(self, tracker):
        self._tracker = tracker
        super().__init__()

    def on_train_begin(self, logs={}):
        self._tracker.update_original_layer_weights(self.model.layers)

    def on_epoch_end(self, epoch, logs={}):
        self._tracker.update_tracker(self.model, logs)

    def on_train_end(self, logs={}):
        with open(constants.TRACKERS_DIRECTORY + "{}.pickle".format(self._tracker.get_name()), 'wb') as pickle_out:
            pickle.dump(self._tracker, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
