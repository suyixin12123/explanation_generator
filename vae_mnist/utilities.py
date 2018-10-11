import tensorflow as tf
from six.moves import urllib
import os
import numpy as np

ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    #expm1 is exponential x minus 1
    return tf.log(tf.expm1(x))

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images



def download(directory, filename):
    """Downloads a file."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = os.path.join(ROOT_PATH, filename)
    print("Downloading %s to %s" % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
    return filepath


def static_mnist_dataset(directory, split_name):
    """Returns binary static MNIST tf.data.Dataset."""
    amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
    dataset = tf.data.TextLineDataset(amat_file)
    str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

    def _parser(s):
        booltensor = tf.py_func(str_to_arr, [s], tf.bool)
        reshaped = tf.reshape(booltensor, [28, 28, 1])
        return tf.to_float(reshaped), tf.constant(0, tf.int32)

    return dataset.map(_parser)


def build_fake_input_fns(IMAGE_SHAPE, batch_size):
    """Builds fake MNIST-style data for unit testing."""
    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.rand(batch_size, *IMAGE_SHAPE).astype("float32")).map(
            lambda row: (row, 0)).batch(batch_size)

    train_input_fn = lambda: dataset.repeat().make_one_shot_iterator().get_next()
    eval_input_fn = lambda: dataset.make_one_shot_iterator().get_next()
    return train_input_fn, eval_input_fn


def build_input_fns(data_dir, batch_size):
    """Builds an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = static_mnist_dataset(data_dir, "train")
    training_dataset = training_dataset.shuffle(50000).repeat().batch(batch_size)
    train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

    # Build an iterator over the heldout set.
    eval_dataset = static_mnist_dataset(data_dir, "valid")
    eval_dataset = eval_dataset.batch(batch_size)
    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

    return train_input_fn, eval_input_fn

