from keras.optimizers import SGD
from regularizers import SquaredFrobReg


# ----- network constants ----- #

CIFAR_INPUT_SHAPE = (32, 32, 3)
IMAGENET_INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 128
IMAGENET_EPOCH_SIZE = 1280000
IMAGENET_VALIDATION_SIZE = 50000
CIFAR_NUM_OF_CLASSES = 10
IMAGENET_NUM_OF_CLASSES = 1000
NUM_OF_CONV_LAYERS = 7
NUM_OF_CHANNELS = 128
KERNEL_SIZE = 5
NUM_OF_FC_LAYERS = 2
FC_SIZE = 256
REGULARIZER = SquaredFrobReg
REGULARIZATION_COEFF = 2
CONV_ACTIVATION = 'relu'
FC_ACTIVATION = 'relu'
LAST_ACTIVATION = 'softmax'
LEARNING_RATE = 0.001
OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=False)
LOSS = 'categorical_crossentropy'
EPOCHS = 150
EXAMPLES_FOR_APPROX = 5000


# ----- other constants ----- #

RESULTS_DIR = '/cs/usr/nadavsch/Desktop/norm_bound_project/results/'
TRACKERS_DIRECTORY = '/cs/usr/nadavsch/Desktop/norm_bound_project/trackers/'
MODELS_DIRECTORY = '/cs/usr/nadavsch/Desktop/norm_bound_project/models/'

IMAGENET_TRAIN_DIR = '/mnt/local-ssd/nadavsch/ImageNet/train_tfr_processed/'
IMAGENET_VAL_DIR = '/mnt/local-ssd/nadavsch/ImageNet/val_tfr_processed/'
