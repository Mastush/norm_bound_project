import keras
import pickle
import constants
from visualization import Visualizer
import utils
import os

# tracker_name = 'baseline_model_reg_None_coeff_0.0_18_april.pickle'
# for tracker_name in os.listdir(constants.TRACKERS_DIRECTORY):
#     with open(constants.TRACKERS_DIRECTORY + tracker_name, 'rb') as pickle_in:
#         tracker = pickle.load(pickle_in)
#     visualizer = Visualizer(tracker, target_directory=constants.RESULTS_DIR + utils.clip_filename(tracker_name))
#     jump = 1
#     visualizer.plot_distances_from_initialization(jump=jump)
#     visualizer.plot_normalized_distances_from_initialization(jump=jump)
#     visualizer.plot_generalization_error(jump=jump)
#     visualizer.plot_accuracy(jump=jump)


from keras.preprocessing.image import ImageDataGenerator
import time
#
for batch_size in [32, 64, 128, 256]:
    start_time = time.time()
    training_gen = ImageDataGenerator()
    training_gen = training_gen.flow_from_directory('/mnt/local-hd/nadavsch/ImageNet2012/train',
                                                    target_size=(224, 224), batch_size=batch_size)
    print("Time for setting generator up: {}".format(time.time() - start_time))

    start_time = time.time()
    for i in range(10):
        a = next(training_gen)
    print("Average time for batch of size {}: {}".format(batch_size, (time.time() - start_time) / 10))


print("------------------- Now mine ------------------")


from data_generator import get_generator

for batch_size in [32, 64, 128, 256]:
    start_time = time.time()
    nadav = get_generator('/mnt/local-hd/nadavsch/ImageNet2012/train', batch_size)
    print("Time for setting generator up: {}".format(time.time() - start_time))

    start_time = time.time()
    for i in range(10):
        a = next(nadav)
    print("Average time for batch of size {}: {}".format(batch_size, (time.time() - start_time) / 10))