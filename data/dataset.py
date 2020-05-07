import csv
import json
import random

import tensorflow as tf

tf.random.set_random_seed(1950)
random.seed(1950)

IMG_SIZE = 160
IMG_SIZE_ORIG = 190
FETCH_SIZE = 16
BUFFER_SIZE = 4096

GRAPH_PB_PATH = 'predictor/predictor.pb'
INPUT_LAYER = 'recog/input:0'
TRAIN_PHASE = 'recog/phase_train:0'

config = json.load(open('data/config.json'))


class Dataset():
    def __init__(self, data_file, data_path, extract_layer, batch_size=32, test=False):
        all_image_paths = []
        all_image_labels = []
        self.size = len(open(data_file).readlines())
        self.batch_size = batch_size
        self.test = test
        self.extract_layer = extract_layer
        with open(data_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=':')
            for row in csv_reader:
                image_path, yaw, pitch, roll = data_path + row[0], float(row[1]), float(row[2]), float(row[3])
                all_image_paths.append(image_path)
                all_image_labels.append([yaw, pitch, roll])

        images_paths = tf.data.Dataset.from_tensor_slices(all_image_paths)
        angles = tf.data.Dataset.from_tensor_slices(all_image_labels)
        self.data = tf.data.Dataset.zip((images_paths, angles))

        self.data = self.data.map(self.load_and_preprocess_image, num_parallel_calls=4)
        if not test: self.data = self.data.shuffle(buffer_size=BUFFER_SIZE)
        self.data = self.data.batch(batch_size).prefetch(buffer_size=FETCH_SIZE)
        self.iterator = self.data.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

    def batches_per_epoch(self):
        return self.size // self.batch_size

    def load_and_preprocess_image(self, path, angles):
        return Dataset._load_image_tf(path, test=self.test), angles

    def initialize(self, session):
        session.run(self.iterator.initializer)

    def get_batch(self, session):
        try:
            image, angles = session.run(self.next_element)
            image = session.run(self.extract_layer, feed_dict={INPUT_LAYER: image, TRAIN_PHASE: False})
        except Exception as e:
            image, angles = None, None

        return image, angles

    @staticmethod
    def _load_image_tf(image_path, test=False):
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if not test:
            image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
            image = Dataset._color_augumentation(image)
        else:
            image = tf.image.central_crop(image, IMG_SIZE / IMG_SIZE_ORIG)
        mean = tf.math.reduce_mean(image)
        std = tf.math.reduce_std(image)
        std_adj = tf.math.maximum(std, 1.0 / tf.math.sqrt(tf.dtypes.cast(tf.size(image), dtype=tf.float32)))
        y = tf.multiply(tf.subtract(image, mean), 1. / std_adj)
        return y

    @staticmethod
    def _color_augumentation(image):
        if random.random() < 0.2: image = tf.image.random_brightness(image, max_delta=0.4)
        if random.random() < 0.2: image = tf.image.random_hue(image, max_delta=0.4)
        if random.random() < 0.2: image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        if random.random() < 0.2: image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        return image


def load_inception_model(session, path=GRAPH_PB_PATH):
    from tensorflow.python.platform import gfile
    with gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        session.graph.as_default()
        tf.import_graph_def(graph_def, name="recog")


def get_train_datasets(machine, model, train_set, val_set, layer, batch, session):
    if model == "inception":
        load_inception_model(session)
    else:
        raise Exception("Model does not exist!")

    train_data = config[machine][f"{train_set}_{model}"]
    val_data = config[machine][f"{val_set}_{model}"]

    dataset_train = Dataset(train_data["csv"], train_data["data"], layer, batch)
    dataset_val = Dataset(val_data["csv"], val_data["data"], layer, batch, test=True)
    return dataset_train, dataset_val


def get_train_test_datasets(machine, model, train_set, val_set, test_set, layer, batch, session):
    if model == "inception":
        load_inception_model(session)
    else:
        raise Exception("Model does not exist!")

    train_data = config[machine][f"{train_set}_{model}"]
    val_data = config[machine][f"{val_set}_{model}"]
    test_data = config[machine][f"{test_set}_{model}"]

    dataset_train = Dataset(train_data["csv"], train_data["data"], layer, batch)
    dataset_val = Dataset(val_data["csv"], val_data["data"], layer, batch, test=True)
    dataset_test = Dataset(test_data["csv"], test_data["data"], layer, batch, test=True)
    return dataset_train, dataset_val, dataset_test


def get_test_dataset(machine, model, test_set, layer, batch, session):
    if model == "inception":
        load_inception_model(session)
    else:
        raise Exception("Model does not exist!")

    test_data = config[machine][f"{test_set}_{model}"]

    dataset_test = Dataset(test_data["csv"], test_data["data"], layer, batch, test=True)
    return dataset_test
