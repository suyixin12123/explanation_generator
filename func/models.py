from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
import numpy as np
from six.moves import urllib
import tensorflow as tf
import tensorflow_probability as tfp
import utilities as ut 
tfd = tfp.distributions

        
def make_encoder(activation, latent_size, base_depth):
    """Creates the encoder function.

    Args:
        activation: Activation function in hidden layers.
        latent_size: The dimensionality of the encoding.
        base_depth: The lowest depth for a layer.

    Returns:
        encoder: A `callable` mapping a `Tensor` of images to a
        `tfd.Distribution` instance over encodings.
    """
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    encoder_net = tf.keras.Sequential([
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(4 * latent_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2 * latent_size, activation=None),
    ])
    #output the 2 * latent_size because that first half used to generate mean, 
    # the second half used to generate diagonic covariance matrix

    def encoder(images):
        images = 2 * tf.cast(images, dtype=tf.float32) - 1
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                    ut._softplus_inverse(1.0)),
            name="code")
    #encoder returns a multivariate normal distribution
    return encoder


def make_decoder(activation, latent_size, output_shape, base_depth):
    """Creates the decoder function.

    Args:
        activation: Activation function in hidden layers.
        latent_size: Dimensionality of the encoding.
        output_shape: The output image shape.
        base_depth: Smallest depth for a layer.

    Returns:
        decoder: A `callable` mapping a `Tensor` of encodings to a
        `tfd.Distribution` instance over images.
    """
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    decoder_net = tf.keras.Sequential([
        deconv(2 * base_depth, 7, padding="VALID"),
        deconv(2 * base_depth, 5),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5),
        conv(output_shape[-1], 5, activation=None),
    ])

    def decoder(codes):
        original_shape = tf.shape(codes)
        # Collapse the sample and batch dimension and convert to rank-4 tensor for
        # use with a convolutional decoder network.
        codes = tf.reshape(codes, (-1, 1, 1, latent_size))
        logits = decoder_net(codes)
        logits = tf.reshape(
            logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
        return tfd.Independent(tfd.Bernoulli(logits=logits),
                            reinterpreted_batch_ndims=len(output_shape),
                            name="image")

    return decoder

def make_mixture_prior(latent_size, mixture_components):
    """Creates the mixture of Gaussians prior distribution.

    Args:
        latent_size: The dimensionality of the latent representation.
        mixture_components: Number of elements of the mixture.

    Returns:
        random_prior: A `tfd.Distribution` instance representing the distribution
        over encodings in the absence of any evidence.
    """
    if mixture_components == 1:
        # See the module docstring for why we don't learn the parameters here.
        return tfd.MultivariateNormalDiag(
            loc=tf.zeros([latent_size]),
            scale_identity_multiplier=1.0)

    loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
    raw_scale_diag = tf.get_variable(
        name="raw_scale_diag", shape=[mixture_components, latent_size])
    mixture_logits = tf.get_variable(
        name="mixture_logits", shape=[mixture_components])

    return tfd.MixtureSameFamily(
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=tf.nn.softplus(raw_scale_diag)),
        mixture_distribution=tfd.Categorical(logits=mixture_logits),
        name="prior")

def make_classifier_mlp(activation, latent_size, output_class_num):
    """Creates the classifier function using mlp model, it will
        use the output of the encoder as the input.

    Args:
        activation: Activation function in hidden layers.
        latent_size: Dimensionality of the encoding.
        output_class: The output image shape.

    Returns:
        classifier: A `callable` mapping a `Tensor` of encodings to a
        logits with dimension of number of class. 
    """
    classifier_net = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(10, activation=None),
    ])
 
    def classifier(codes):
        logits = classifier_net(codes)
        return logits
    return classifier 


