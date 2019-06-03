#!/cs/usr/bin/python3.5


import model_creation
from constants import *
from metrics import get_metrics
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from callbacks import TrackerCallback, get_non_tracker_callbacks
from tracker import Tracker
import time
from data_generator import ImageNetTFRecordDataset



def cifar_experiments():
    # prepare database
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_val, y_val = x_test[:len(x_test)//2], y_test[:len(y_test)//2]
    x_test, y_test = x_test[len(x_test)//2:], y_test[len(y_test)//2:]
    y_train, y_val, y_test = to_categorical(y_train, CIFAR_NUM_OF_CLASSES), \
                             to_categorical(y_val, CIFAR_NUM_OF_CLASSES), to_categorical(y_test, CIFAR_NUM_OF_CLASSES)

    x_train, y_train = x_train[:600], y_train[:600]
    x_val, y_val = x_val[:10], y_val[:10]

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # get generator with augmentations
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)

    # Prepare experiments
    iterations_without_reg = 1
    iterations_with_distance_reg = 10
    iterations_with_l2_reg = 10
    initial_coeff = 0.0001
    regularizers_list = [None for _ in range(iterations_without_reg)] + \
                        [SquaredFrobReg for _ in range(iterations_with_distance_reg)] + \
                        [l2 for _ in range(iterations_with_l2_reg)]
    coeff_list = [0.0 for _ in range(iterations_without_reg)] + \
                 [initial_coeff * 2**i for i in range(iterations_with_distance_reg)] + \
                 [initial_coeff * 2**i for i in range(iterations_with_l2_reg)]
    num_of_experiments = iterations_without_reg + iterations_with_distance_reg + iterations_with_l2_reg

    # now prepare and perform each experiment

    start_time = time.time()

    for i in range(num_of_experiments):
        last_time = time.time()
        regularizer = regularizers_list[i]
        regularizer_name = "None" if regularizer is None else regularizer.__name__
        coeff = coeff_list[i]

        # get model
        model = model_creation.get_convolutional_model(CIFAR_INPUT_SHAPE, CIFAR_NUM_OF_CLASSES, NUM_OF_CONV_LAYERS,
                                                       NUM_OF_CHANNELS, KERNEL_SIZE, 1, NUM_OF_FC_LAYERS, 200, regularizer,
                                                       coeff, CONV_ACTIVATION, FC_ACTIVATION,
                                                       LAST_ACTIVATION, OPTIMIZER, LOSS, get_metrics(),
                                                       padding='valid')
        model.summary()

        # arrange callbacks
        model_name = "reg_{}_coeff_{}_25_may".format(regularizer_name, coeff)
        print("Model name is: {}".format(model_name))
        tracker = Tracker(model_name, x_train, y_train, x_val, y_val, compute_training_error=True,
                          examples_for_training_error_approx=None, save=True)

        callbacks = get_non_tracker_callbacks(model_name) + [TrackerCallback(tracker)]

        # fit
        # model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        #                     steps_per_epoch=x_train.shape[0] // BATCH_SIZE, epochs=EPOCHS,
        #                     callbacks=callbacks, validation_data=(x_val, y_val), verbose=1)

        model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                            steps_per_epoch=5, epochs=EPOCHS,
                            callbacks=callbacks, validation_data=(x_val, y_val), verbose=1)

        # indicate end of experiment
        experiment_time = int(time.time() - last_time) / 60
        total_time = int(time.time() - start_time) / 60
        print("Finished experiment number {} of model {}. \nThis Experiment took {:.2f} minutes. "
              "Elapsed time: {:.2f} minutes.".format(i + 1, model_name, experiment_time, total_time))


def imagenet_experiments():
    # get generators
    train_dataset = ImageNetTFRecordDataset(IMAGENET_TRAIN_DIR, BATCH_SIZE)
    val_dataset = ImageNetTFRecordDataset(IMAGENET_VAL_DIR, BATCH_SIZE)

    # train_gen = ImageDataGenerator(samplewise_center=False, samplewise_std_normalization=False)
    # val_gen = ImageDataGenerator(samplewise_center=False, samplewise_std_normalization=False)
    #
    # train_gen = train_gen.flow_from_directory('/mnt/local-hd/nadavsch/ImageNet2012/train_processed',
    #                                           target_size=(224, 224), batch_size=BATCH_SIZE)
    # val_gen = val_gen.flow_from_directory('/mnt/local-hd/nadavsch/ImageNet2012/val_processed',
    #                                       target_size=(224, 224), batch_size=BATCH_SIZE)

    # Prepare experiments
    iterations_without_reg = 1
    iterations_with_distance_reg = 10
    iterations_with_l2_reg = 10
    initial_coeff = 0.0001
    regularizers_list = [None for _ in range(iterations_without_reg)] + \
                        [SquaredFrobReg for _ in range(iterations_with_distance_reg)] + \
                        [l2 for _ in range(iterations_with_l2_reg)]
    coeff_list = [0.0 for _ in range(iterations_without_reg)] + \
                 [initial_coeff * 2**i for i in range(iterations_with_distance_reg)] + \
                 [initial_coeff * 2**i for i in range(iterations_with_l2_reg)]
    num_of_experiments = iterations_without_reg + iterations_with_distance_reg + iterations_with_l2_reg

    # now prepare and perform each experiment

    start_time = time.time()

    for i in range(num_of_experiments):
        last_time = time.time()
        regularizer = regularizers_list[i]
        regularizer_name = "None" if regularizer is None else regularizer.__name__
        coeff = coeff_list[i]

        # get tensors
        x, y = train_dataset.create_dataset()
        x_val, y_val = val_dataset.create_dataset()

        # get model
        model = model_creation.get_convolutional_model(IMAGENET_INPUT_SHAPE, IMAGENET_NUM_OF_CLASSES, 6,
                                                       [128, 128, 128, 128, 128, 128],
                                                       [11, 5, 5, 5, 5, 5], [4, 2, 1, 1, 2, 1], 2,
                                                       400, regularizer, coeff, CONV_ACTIVATION, FC_ACTIVATION,
                                                       LAST_ACTIVATION, OPTIMIZER, LOSS, get_metrics(),
                                                       padding='valid', batch_norm=False, tensors=[x, y])

        # from keras.applications.resnet50 import ResNet50
        #
        # model = ResNet50(weights=None, input_tensor=x)
        # model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=get_metrics(), target_tensors=[y])

        # model = model_creation.alexnet(x, y)

        model.summary()

        # arrange callbacks
        model_name = "reg_{}_coeff_{}_25_may".format(regularizer_name, coeff)
        print("Model name is: {}".format(model_name))
        tracker = Tracker(model_name, None, None, None, None, compute_training_error=False,
                          examples_for_training_error_approx=None, save=True)

        callbacks = get_non_tracker_callbacks(model_name) + [TrackerCallback(tracker)]

        # fit
        model.fit(epochs=EPOCHS, verbose=1, validation_data=(x_val, y_val),
                  steps_per_epoch=IMAGENET_EPOCH_SIZE // BATCH_SIZE,
                  validation_steps=IMAGENET_VALIDATION_SIZE // BATCH_SIZE)

        # model.fit_generator(train_gen, steps_per_epoch=IMAGENET_EPOCH_SIZE // BATCH_SIZE, epochs=EPOCHS,
        #                     validation_data=val_gen, validation_steps=IMAGENET_VALIDATION_SIZE // BATCH_SIZE,
        #                     workers=10, use_multiprocessing=True)

        # indicate end of experiment
        experiment_time = int(time.time() - last_time) / 60
        total_time = int(time.time() - start_time) / 60
        print("Finished experiment number {} of model {}. \nThis Experiment took {:.2f} minutes. "
              "Elapsed time: {:.2f} minutes.".format(i + 1, model_name, experiment_time, total_time))


if __name__ == '__main__':
    cifar_experiments()
    imagenet_experiments()

