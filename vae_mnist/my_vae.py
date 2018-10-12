
# my vae leraning model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from tensorflow.app import flags
#from absl import flags
import numpy as np
import tensorflow as tf
import utilities as ut
from vae_model_fn import vae 

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "latent_size",
    default=16,
    help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_integer(
    "mixture_components",
    default=100,
    help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")
flags.DEFINE_bool(
    "analytic_kl",
    default=False,
    help="Whether or not to use the analytic version of the KL. When set to "
         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
         "then you must also specify `mixture_components=1`.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=5, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "fake_data",
    default=False,
    help="If true, uses fake data instead of MNIST.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS

def main(argv):
    del argv  # unused
    print("begin to training...")
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
        tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    print("preparing data...")
    if FLAGS.fake_data:
        train_input_fn, eval_input_fn = ut.build_fake_input_fns(IMAGE_SHAPE, FLAGS.batch_size)
    else:
        train_input_fn, eval_input_fn = ut.build_input_fns(FLAGS.data_dir,
                                                        FLAGS.batch_size)
    print("building the model...")
    vae_model = vae(IMAGE_SHAPE)
    model_fn = vae_model.model_fn
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_steps=FLAGS.viz_steps,
        ),
    )

    for i in range(FLAGS.max_steps // FLAGS.viz_steps):
        print("training the round:", i*FLAGS.viz_steps)
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)

    print("evaluating...")
    eval_results = estimator.evaluate(eval_input_fn)
    print("Evaluation_results:\n\t%s\n" % eval_results)

if __name__ == "__main__":
    tf.app.run()






