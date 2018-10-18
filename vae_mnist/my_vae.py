
# my vae leraning model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os, sys
sys.path.append(os.path.relpath("../func"))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
<<<<<<< HEAD
    "viz_steps", default=100, help="Frequency at which to save visualizations.")
=======
    "viz_steps", default=200, help="Frequency at which to save visualizations.")
>>>>>>> 50aab253f3a7946ae38d7be69e1d5a4fcd65b798
flags.DEFINE_bool(
    "fake_data",
    default=False,
    help="If true, uses fake data instead of MNIST.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")
flags.DEFINE_string(
    "dataset",
    default="fasion_mnist",
    help="choose dataset to train, current support:  \
          *. mnist    \
          *. fasion_mnist")

FLAGS = flags.FLAGS

def main(argv):
    del argv  # unused
    print("begin to training...")
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    train_input_fn, eval_input_fn = ut.preparing_data_image(FLAGS)
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






