from models import make_encoder, make_decoder, make_mixture_prior
import utilities as ut
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class vae:

    def __init__(self, IMAGE_SHAPE):
        self.IMAGE_SHAPE = IMAGE_SHAPE

    def image_tile_summary(self, name, tensor, rows=8, cols=8):
        tf.summary.image(name, ut.pack_images(tensor, rows, cols), max_outputs=1)


    def model_fn(self, features, labels, mode, params, config):
        """Builds the model function for use in an estimator.

        Arguments:
            features: The input features for the estimator.
            labels: The labels, unused here.
            mode: Signifies whether it is train or test or predict.
            params: Some hyperparameters as a dictionary.
            config: The RunConfig, unused here.

        Returns:
            EstimatorSpec: A tf.estimator.EstimatorSpec instance.
        """
        del labels, config

        if params["analytic_kl"] and params["mixture_components"] != 1:
            raise NotImplementedError(
                "Using `analytic_kl` is only supported when `mixture_components = 1` "
                "since there's no closed form otherwise.")

        encoder = make_encoder(params["activation"],
                                params["latent_size"],
                                params["base_depth"])
        decoder = make_decoder(params["activation"],
                                params["latent_size"],
                                self.IMAGE_SHAPE,
                                params["base_depth"])
        latent_prior = make_mixture_prior(params["latent_size"],
                                            params["mixture_components"])

        self.image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

        decoder_likelihood = decoder(latent_prior.sample(params["n_samples"]))
        decoder_likelihood_sample = decoder_likelihood.sample(params["n_samples"])

        encoder_posterior_fake = encoder(decoder_likelihood_sample)
        encoder_posterior_real = encoder(features)
        self.image_tile_summary(
            "recon/sample",
            tf.to_float(decoder_likelihood.sample()[:3, :16]),
            rows=3,
            cols=16)
        self.image_tile_summary(
            "recon/mean",
            decoder_likelihood.mean()[:3, :16],
            rows=3,
            cols=16)

        # `distortion` is just the negative log likelihood.
        #distortion = -decoder_likelihood.log_prob(features)
        #avg_distortion = tf.reduce_mean(distortion)
        tf.summary.scalar("distortion", avg_distortion)
        real_fake_rate = tfd.kl_divergence(encoder_posterior_fake, encoder_posterior_real)
        if params["analytic_kl"]:
            rate = tfd.kl_divergence(encoder_posterior_real, latent_prior)
        else:
            encoder_posterior_real_sample = encoder_posterior_real.sample(params["n_samples"])
            rate = (encoder_posterior_real.log_prob(encoder_posterior_real_sample)
                    - latent_prior.log_prob(encoder_posterior_real_sample))
        
        avg_rf_rate = tf.reduce_mean(real_fake_rate)
        tf.summary.scalar("real_fake_rate", avg_rf_rate)
        avg_rate = tf.reduce_mean(rate)
        tf.summary.scalar("rate", avg_rate)

        elbo_local = -(rate + real_fake_rate)

        elbo = tf.reduce_mean(elbo_local)
        loss = -elbo
        tf.summary.scalar("elbo", elbo)

        importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(params["n_samples"])))
        tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

        # Decode samples from the prior for visualization.
        random_image = decoder(latent_prior.sample(16))
        self.image_tile_summary(
            "random/sample", tf.to_float(random_image.sample()), rows=4, cols=4)
        self.image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)

        # Perform variational inference by minimizing the -ELBO.
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                                params["max_steps"])
        tf.summary.scalar("learning_rate", learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={
                "elbo": tf.metrics.mean(elbo),
                "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
                "rate": tf.metrics.mean(avg_rate),
                "real_fake_rate": tf.metrics.mean(avg_rf_rate),
            },
        )


