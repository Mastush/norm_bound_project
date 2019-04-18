#!/cs/usr/bin/python3.5


import model_creation
from constants import *
from metrics import get_metrics
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from callbacks import TrackerCallback, get_non_tracker_callbacks
from tracker import Tracker


if __name__ == "__main__":

    # prepare database
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = to_categorical(y_train, NUM_OF_CLASSES), to_categorical(y_test, NUM_OF_CLASSES)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # get generator with augmentations
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # Prepare experiments
    iterations_without_reg = 1
    iterations_with_distance_reg = 10
    iterations_with_l2_reg = 10
    coeff_min = 0.01
    coeff_jump = 0.02
    regularizers_list = [None for _ in range(iterations_without_reg)] + \
                        [SquaredFrobReg for _ in range(iterations_with_distance_reg)] + \
                        [l2 for _ in range(iterations_with_l2_reg)]
    coeff_list = [0.0 for _ in range(iterations_without_reg)] + \
                 [coeff_min + (coeff_jump * i) for i in range(iterations_with_distance_reg)] + \
                 [coeff_min + (coeff_jump * i) for i in range(iterations_with_l2_reg)]
    num_of_experiments = iterations_without_reg + iterations_with_distance_reg + iterations_with_l2_reg

    # now prepare and perform each experiment
    for i in range(num_of_experiments):
        regularizer = regularizers_list[i]
        regularizer_name = "None" if regularizer == None else regularizer.__name__
        coeff = coeff_list[i]

        # get model
        model = model_creation.get_convolutional_model(INPUT_SHAPE, NUM_OF_CLASSES, NUM_OF_CONV_LAYERS, NUM_OF_CHANNELS,
                                                       KERNEL_SIZE, NUM_OF_FC_LAYERS, FC_SIZE, regularizer,
                                                       coeff, CONV_ACTIVATION, FC_ACTIVATION,
                                                       LAST_ACTIVATION, OPTIMIZER, LOSS, get_metrics())
        model.summary()

        # arrange callbacks
        model_name = "baseline_model_reg_{}_coeff_{}_18_april".format(regularizer_name, coeff)
        print("Model name is: {}".format(model_name))
        tracker = Tracker(model_name)
        callbacks = get_non_tracker_callbacks(model_name) + [TrackerCallback(tracker)]

        # fit
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                            steps_per_epoch=x_train.shape[0] // BATCH_SIZE, epochs=EPOCHS,
                            callbacks=callbacks, validation_data=(x_test, y_test))

        # indicate end of experiment
        print("Finished experiment number {} of model {}".format(i + 1, model_name))