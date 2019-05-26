from keras.callbacks import Callback, ModelCheckpoint
import constants
import time


class TrackerCallback(Callback):

    def __init__(self, tracker):
        self._tracker = tracker
        super().__init__()

    def on_train_begin(self, logs={}):
        self._tracker.update_original_layer_weights(self.model.layers)

    def on_epoch_end(self, epoch, logs={}):
        start_time = time.time()
        self._tracker.update_tracker(self.model, logs)
        print("Tracker callback took {:.2f} seconds.".format(time.time() - start_time))

    def on_train_end(self, logs={}):
        self._tracker.update_last_layer_weights(self.model.layers)
        self._tracker.save_tracker()


def get_non_tracker_callbacks(model_name=""):
    filepath = constants.MODELS_DIRECTORY + model_name + "_{epoch:02d}_{val_categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)
    # return [checkpoint]
    return []
