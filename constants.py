from keras.optimizers import SGD
from regularizers import SquaredFrobReg


# ----- network constants ----- #

INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 64
NUM_OF_CLASSES = 10
NUM_OF_CONV_LAYERS = 5
NUM_OF_CHANNELS = 64
KERNEL_SIZE = 3
NUM_OF_FC_LAYERS = 2
FC_SIZE = 128
REGULARIZER = SquaredFrobReg
# REGULARIZER = None
REGULARIZATION_COEFF = 1
CONV_ACTIVATION = 'relu'
FC_ACTIVATION = 'relu'
LAST_ACTIVATION = 'softmax'
LEARNING_RATE = 0.001
OPTIMIZER = SGD(lr=LEARNING_RATE)
LOSS = 'categorical_crossentropy'
EPOCHS = 150


# ----- other constants ----- #

TRACKERS_DIRECTORY = '/cs/usr/nadavsch/Desktop/norm_bound_project/trackers/'
MODELS_DIRECTORY = '/cs/usr/nadavsch/Desktop/norm_bound_project/models/'