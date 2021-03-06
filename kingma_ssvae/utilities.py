import tensorflow as tf
import tensorflow_probability as tfp
from six.moves import urllib
import os
import numpy as np
from sklearn.utils import shuffle

tfd = tfp.distributions

ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def preparing_data_image(FLAGS):

    dataset_dict = {}
    dataset_dict["fashion_mnist"] = tf.keras.datasets.fashion_mnist
    dataset_dict['mnist'] = tf.keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = \
            dataset_dict[FLAGS.dataset].load_data()
    
    train_images = tf.manip.reshape(train_images, [-1, 28, 28, 1]) /255
    test_images = tf.manip.reshape(test_images, [-1, 28, 28, 1]) /255
    training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    training_dataset = training_dataset.shuffle(50000).repeat().batch(FLAGS.batch_size)
    train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(FLAGS.batch_size) 
    test_input_fn = lambda: test_dataset.make_one_shot_iterator().get_next()

    return train_input_fn, test_input_fn

def preparing_data_image_semi(FLAGS):
    seed = 50000
    dataset_dict = {}
    dataset_dict["fashion_mnist"] = tf.keras.datasets.fashion_mnist
    dataset_dict['mnist'] = tf.keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = \
            dataset_dict[FLAGS.dataset].load_data()

    train_num = len(train_images)
    train_labeled_num = int(train_num * FLAGS.labeled_data_rate)

    train_images, train_labels = \
        shuffle(train_images, train_labels, random_state=seed)

    train_images = tf.manip.reshape(train_images, [-1, 28, 28, 1]) /255

    train_images_labeled = train_images[:train_labeled_num]
    train_labels_labeled = train_labels[:train_labeled_num] 

    train_images_unlabeled = train_images[train_labeled_num:]

    #unused here.
    #train_labels_unlabeled = train_labels[train_labeled_num:]

    training_dataset_labeled_image = \
        tf.data.Dataset.from_tensor_slices((train_images_labeled))
    training_dataset_labeled_image = \
        training_dataset_labeled_image.repeat().batch(FLAGS.batch_size)

    training_dataset_unlabeled_image = \
        tf.data.Dataset.from_tensor_slices((train_images_unlabeled))
    training_dataset_unlabeled_image = \
        training_dataset_unlabeled_image.repeat().batch(FLAGS.batch_size)
    
    train_dataset_label = \
        tf.data.Dataset.from_tensor_slices((train_labels_labeled))
    train_dataset_label = \
        train_dataset_label.repeat().batch(FLAGS.batch_size)
    

    train_labeled_image_fn = training_dataset_labeled_image.make_one_shot_iterator()
    train_unlabeled_image_fn = training_dataset_unlabeled_image.make_one_shot_iterator()
    train_labeled_labels_fn = train_dataset_label.make_one_shot_iterator()


    test_images = tf.manip.reshape(test_images, [-1, 28, 28, 1]) /255
    test_dataset_image = tf.data.Dataset.from_tensor_slices(test_images)
    test_dataset_label = tf.data.Dataset.from_tensor_slices(test_labels)
    test_dataset_image = test_dataset_image.batch(FLAGS.batch_size) 
    test_dataset_label = test_dataset_label.batch(FLAGS.batch_size) 

    test_image_fn = test_dataset_image.make_one_shot_iterator()
    test_label_fn = test_dataset_label.make_one_shot_iterator()

    return train_labeled_image_fn, train_unlabeled_image_fn, train_labeled_labels_fn, test_image_fn, test_label_fn



def preparing_data_mnist(IMAGE_SHAPE, FLAGS):
    if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
        tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    print("preparing data...")
    if FLAGS.fake_data:
        train_input_fn, eval_input_fn = build_fake_input_fns(IMAGE_SHAPE, FLAGS.batch_size)
    else:
        train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir,
                                                        FLAGS.batch_size)
 
    return train_input_fn, eval_input_fn

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


def gen_eval_samples(eval_posterior, latent_size): 
    """
    FOR beta_VAE use to sample eval samples
    Generate evaluation samples for decoder 
    eval_posterior: tfd.MultivariateDiag distribution with \ 
                    (1, latent size) 
    return: tfd.MultivariateDiag distribution that changes each 
            dimension 10 times respectively 
    """
    loc = eval_posterior.loc
    var = eval_posterior.variance()
    if len(loc.shape) == 1:
        loc = tf.expand_dims(loc, 0)
        var = tf.expand_dims(var, 0)
        
    new_var = tf.manip.tile(var, [10*latent_size, 1])
    new_loc = tf.manip.tile(loc, [10*latent_size, 1])
    loc_modify = np.array([[0.1*(i%10)-0.5 if int(i/10)==j else 0 for j in \
                range(latent_size)] for i in range(latent_size*10)])
    
    new_loc = new_loc + loc_modify
    #new_loc = tf.Print(new_loc, [new_loc], "evaluation loc")
    new_distribution = tfd.MultivariateNormalDiag(
        loc= new_loc,
        scale_diag= new_var,
        name="eval_code")

    return new_distribution.sample()


