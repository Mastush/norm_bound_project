#!/cs/usr/bin/python3.5


import model_creation
from constants import *
from metrics import get_metrics
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from callbacks import TrackerCallback, get_non_tracker_callbacks
from tracker import Tracker

if __name__ == "__main__":

    # prepare database
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train, y_train = x_train[:500], y_train[:500]
    y_train, y_test = to_categorical(y_train, NUM_OF_CLASSES), to_categorical(y_test, NUM_OF_CLASSES)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # get model
    model = model_creation.get_convolutional_model(INPUT_SHAPE, NUM_OF_CLASSES, NUM_OF_CONV_LAYERS, NUM_OF_CHANNELS,
                                                   KERNEL_SIZE, NUM_OF_FC_LAYERS, FC_SIZE, REGULARIZER,
                                                   REGULARIZATION_VECTOR, CONV_ACTIVATION, FC_ACTIVATION,
                                                   LAST_ACTIVATION, OPTIMIZER, LOSS, get_metrics())
    model.summary()

    # get augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # arrange callbacks
    model_name = "baseline_model_with_reg_coeff_{}_15_april".format(REGULARIZATION_VECTOR)
    tracker = Tracker(model_name)
    callbacks = get_non_tracker_callbacks(model_name) + [TrackerCallback(tracker)]

    # fit
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=x_train.shape[0] // BATCH_SIZE, epochs=100,
                        callbacks=callbacks, validation_data=(x_test, y_test))
