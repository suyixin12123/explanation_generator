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
        #conv(base_depth, 5, 1),
        #conv(base_depth, 5, 2),
        #conv(2 * base_depth, 5, 1),
        #conv(2 * base_depth, 5, 2),
        #conv(4 * latent_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=activation),
        tf.keras.layers.Dense(2 * latent_size, activation=None),
    ])
    #output the 2 * latent_size because that first half used to generate mean, 
    # the second half used to generate diagonic covariance matrix

    def encoder(images):
        images = tf.cast(images, dtype=tf.float32)
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                    ut._softplus_inverse(1.0)),
            name="latent_representation")
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


       
def make_encoder_joint_input(activation, latent_size, base_depth):
    """Creates the encoder function.

    Args:
        activation: Activation function in hidden layers.
        latent_size: The dimensionality of the encoding.
        base_depth: The lowest depth for a layer.
        encoder: A `callable` mapping a `Tensor` of images to a
        `tfd.Distribution` instance over encodings.
    """

    encoder_net = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(2 * latent_size, activation=None)
    ])

    #codes can be shape of (None,)
    def encoder(images,labels, num_labels):
        images = tf.cast(images, dtype=tf.float32)
        image_input = tf.reshape(images, [-1, 784])
        label_input = tf.one_hot(labels, num_labels)
        encoder_input = tf.concat([image_input, label_input], -1 )
        net = encoder_net(encoder_input)
        return tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                    ut._softplus_inverse(1.0)),
            name="latent_representation")
    #encoder returns a multivariate normal distribution
    return encoder


def make_decoder_joint_input(activation, latent_size, output_shape, base_depth):
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

    decoder_net = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(784, activation=None),
    ])

    def decoder(latent_rep, labels, num_labels):
        original_shape = tf.shape(latent_rep)
        # Collapse the sample and batch dimension and convert to rank-4 tensor for
        # use with a convolutional decoder network.
        latent_rep = tf.reshape(latent_rep, (-1, latent_size))
        labels = tf.one_hot(labels, num_labels)
        labels = tf.reshape(labels, (-1, num_labels))
        
        decoder_input = tf.concat([latent_rep, labels], -1)
        logits = decoder_net(decoder_input)
        logits = tf.reshape(
            logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
        """
        return tfd.Independent(tfd.Normal(loc=logits, scale=1.),
                            reinterpreted_batch_ndims=len(output_shape),
                            name="image")
        """
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

def make_classifier_mlp(activation, latent_size, num_class):
    """Creates the classifier function using mlp model, it will
        use the output of the encoder as the input.

    Args:
        activation: Activation function in hidden layers.
        latent_size: Dimensionality of the encoding.
        output_class: The output image shape.

    Returns:
        classifier: A `callable` mapping a `Tensor` of encodings to a
        Categorical distribution with dimension of number of class. 
    """
    classifier_net = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(num_class, activation=None),
    ])
 
    def classifier(codes):
        logits = classifier_net(codes)
        return tfd.Categorical(
            logits=logits,
            name="code"
        )
    return classifier 


def make_classifier_cnn(activation, latent_size, base_depth, num_class):
    """Creates the classifier function that the input is the 
        picture and the output is class.

    Args:
        activation: Activation function in hidden layers.
        latent_size: The dimensionality of the encoding.
        base_depth: The lowest depth for a layer.

    Returns:
        classifier: A `callable` mapping a `Tensor` of images to a
        number-of-class dimension Categorical distributions over classifiers.
    """
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)
        #tfp.layers.Convolution2DReparameterization, padding="SAME", activation=activation)

    classifier_net = tf.keras.Sequential([
        #tf.keras.layers.Reshape([28,28,1]),
        #conv(base_depth, 5, 1),
        #conv(2 * base_depth, 5, 1),
        #conv(4 * latent_size, 7, padding="VALID"),

        tf.keras.layers.Flatten(),
        #tfp.layers.DenseReparameterization(num_class),
        tf.keras.layers.Dense(num_class),
    ])

    def classifier(images):
        images = tf.cast(images, dtype=tf.float32)
        logits = classifier_net(images)
        prob_logits = tfd.Categorical(
            logits=logits,
            name="code_dist"
        )
        return prob_logits

    return classifier 

