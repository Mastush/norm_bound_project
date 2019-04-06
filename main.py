import model_creation
from constants import *
from metrics import get_metrics
from keras.datasets import cifar10
from keras.utils import to_categorical

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = to_categorical(y_train, NUM_OF_CLASSES), to_categorical(y_test, NUM_OF_CLASSES)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = model_creation.get_convolutional_model(INPUT_SHAPE, NUM_OF_CLASSES, NUM_OF_CONV_LAYERS, NUM_OF_CHANNELS,
                                                   KERNEL_SIZE, NUM_OF_FC_LAYERS, FC_SIZE, REGULARIZER,
                                                   REGULARIZATION_VECTOR, CONV_ACTIVATION, FC_ACTIVATION,
                                                   LAST_ACTIVATION, OPTIMIZER, LOSS, get_metrics())
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=15)