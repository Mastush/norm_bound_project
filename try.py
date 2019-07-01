import model_creation
from constants import *
from metrics import get_metrics
#
# model = model_creation.get_convolutional_model(IMAGENET_INPUT_SHAPE, IMAGENET_NUM_OF_CLASSES, 5, [16, 32, 64, 128, 256],
#                                                3, 2, 3, 500, None, 0, CONV_ACTIVATION, FC_ACTIVATION, LAST_ACTIVATION,
#                                                OPTIMIZER, LOSS, [])
# model.summary()
#
# model = model_creation.get_convolutional_model(IMAGENET_INPUT_SHAPE, IMAGENET_NUM_OF_CLASSES, 4, [32, 64, 128, 256, 512],
#                                                [5, 3, 3, 3, 3], [3, 2, 2, 2, 2], 3, 1000, None, 0, CONV_ACTIVATION, FC_ACTIVATION, LAST_ACTIVATION,
#                                                OPTIMIZER, LOSS, [])
# model.summary()

model = model_creation.get_convolutional_model(IMAGENET_INPUT_SHAPE, IMAGENET_NUM_OF_CLASSES, 5, [64, 128, 256, 512, 1028],
                                               [11, 3, 3, 3, 3], [5, 2, 2, 2, 2], 3, 2000, None, 0, CONV_ACTIVATION, FC_ACTIVATION, LAST_ACTIVATION,
                                               OPTIMIZER, LOSS, get_metrics())
model.summary()
