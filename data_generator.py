import tensorflow as tf
import os


class ImageNetTFRecordDataset:
    def __init__(self, dir, batch_size):
        self._dir = dir
        self._batch_size = batch_size

    @staticmethod
    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                            'image/class/label': tf.FixedLenFeature([], tf.int64)}

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # Turn your saved image string into an array
        parsed_features['image/encoded'] = tf.image.decode_jpeg(parsed_features['image/encoded'])

        parsed_features['image/encoded'] = tf.image.convert_image_dtype(parsed_features['image/encoded'], tf.float32)
        parsed_features['image/encoded'] = tf.image.per_image_standardization(parsed_features['image/encoded'])

        return parsed_features['image/encoded'], parsed_features["image/class/label"]

    def create_dataset(self):

        dir_files = os.listdir(self._dir)
        dataset = tf.data.TFRecordDataset([os.path.join(self._dir, f) for f in dir_files])

        # This dataset will go on forever
        dataset = dataset.repeat()

        # Set the number of datapoints you want to load and shuffle
        dataset = dataset.shuffle(self._batch_size * 10)

        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self._parse_function, num_parallel_calls=8)

        # Set the batchsize
        dataset = dataset.batch(self._batch_size)

        dataset = dataset.prefetch(4)

        # Create an iterator
        iterator = dataset.make_one_shot_iterator()

        # Create your tf representation of the iterator
        image, label = iterator.get_next()

        # Bring your picture back in shape
        image = tf.reshape(image, [-1, 224, 224, 3])

        # Create a one hot array for your labels
        label = tf.one_hot(label, 1000)

        return image, label