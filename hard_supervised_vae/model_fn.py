from models import make_encoder_joint_input, make_decoder_joint_input, make_mixture_prior, make_classifier_cnn
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
            labels: The labels, some of them are used as semisupervised
                    learning.
            mode: Signifies whether it is train or test or predict.
            params: Some hyperparameters as a dictionary.
            config: The RunConfig, unused here.

        Returns:
            EstimatorSpec: A tf.estimator.EstimatorSpec instance.
        """
        del config
        onehot_labels = tf.one_hot(labels, params["num_labels"])

        if params["analytic_kl"] and params["mixture_components"] != 1:
            raise NotImplementedError(
                "Using `analytic_kl` is only supported when `mixture_components = 1` "
                "since there's no closed form otherwise.")

        encoder = make_encoder_joint_input(params["activation"],
                                params["latent_size"],
                                params["base_depth"])
        decoder = make_decoder_joint_input(params["activation"],
                                params["latent_size"],
                                self.IMAGE_SHAPE,
                                params["base_depth"])
        latent_prior = make_mixture_prior(params["latent_size"],
                                            params["mixture_components"])

        self.image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

        approx_posterior = encoder(features, labels, params["num_labels"])
        approx_posterior_sample = approx_posterior.sample(params["n_samples"])
        """
        the first one the input is latent reprentation
        the second one the input is image
        """
        #classifier_logits = classifier(tf.reduce_mean(approx_posterior_sample, 0))
        #code_posterior = classifier(features)
        #code_sample = code_posterior.sample(params["n_samples"])
        #code_sample = tf.one_hot(code_sample, params["num_labels"])
        #code_sample = tf.one_hot(tfd.Categorical(logits=tf.cast(onehot_labels, tf.float32)).sample(params["n_samples"]), params['num_labels'])

        decoder_likelihood = decoder(approx_posterior_sample, \
            labels, params["num_labels"])
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
        distortion = -decoder_likelihood.log_prob(features)
        avg_distortion = tf.reduce_mean(distortion)
        tf.summary.scalar("distortion", avg_distortion)

        if params["analytic_kl"]:
            rate = tfd.kl_divergence(approx_posterior, latent_prior)
        else:
            rate = (approx_posterior.log_prob(approx_posterior_sample)
                    - latent_prior.log_prob(approx_posterior_sample))
        avg_rate = tf.reduce_mean(rate)
        tf.summary.scalar("rate", avg_rate)

        elbo_local = -(params["kl_scalar_param"] * rate + \
                       params["ae_scalar_param"] * distortion)

        elbo = tf.reduce_mean(elbo_local)
        loss = -elbo 
        tf.summary.scalar("elbo", elbo)

        importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(params["n_samples"])))
        tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

        # Decode samples from the prior for visualization.
        random_image = decoder(latent_prior.sample(16), \
            [8 for i in range(16)], params["num_labels"])
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
                "distortion": tf.metrics.mean(avg_distortion),
            },
        )
