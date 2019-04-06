import model_creation
from keras.optimizers import adam
from regularizers import FrobReg

INPUT_SHAPE = (32, 32, 1)
NUM_OF_CLASSES = 10
NUM_OF_CONV_LAYERS = 5
NUM_OF_CHANNELS = 16
KERNEL_SIZE = 3
NUM_OF_FC_LAYERS = 3
FC_SIZE = 64
REGULARIZER = FrobReg
REGULARIZATION_VECTOR = 0
CONV_ACTIVATION = 'relu'
FC_ACTIVATION = 'relu'
LAST_ACTIVATION = 'softmax'
OPTIMIZER = adam()
LOSS = 'categorical_crossentropy'
METRICS = []

if __name__ == "__main__":
    model = model_creation.get_convolutional_model(INPUT_SHAPE, NUM_OF_CLASSES, NUM_OF_CONV_LAYERS, NUM_OF_CHANNELS,
                                                   KERNEL_SIZE, NUM_OF_FC_LAYERS, FC_SIZE, REGULARIZER,
                                                   REGULARIZATION_VECTOR, CONV_ACTIVATION, FC_ACTIVATION,
                                                   LAST_ACTIVATION, OPTIMIZER, LOSS, METRICS)
    model.summary()