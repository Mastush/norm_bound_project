from keras.optimizers import adam
from regularizers import SquaredFrobReg, FrobReg

INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
NUM_OF_CLASSES = 10
NUM_OF_CONV_LAYERS = 5
NUM_OF_CHANNELS = 16
KERNEL_SIZE = 3
NUM_OF_FC_LAYERS = 3
FC_SIZE = 64
REGULARIZER = SquaredFrobReg
REGULARIZATION_VECTOR = 1
CONV_ACTIVATION = 'relu'
FC_ACTIVATION = 'relu'
LAST_ACTIVATION = 'softmax'
OPTIMIZER = adam()
LOSS = 'categorical_crossentropy'