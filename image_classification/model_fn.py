from models import make_encoder, make_decoder_joint_input, make_mixture_prior, make_classifier_cnn
import utilities as ut
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class classifier:

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

        classifier = make_classifier_cnn(tf.nn.relu,
                                params["latent_size"],
                                params["base_depth"],
                                params["num_labels"])

        self.image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

        logits_prob = classifier(features)
        logits = logits_prob.sample()
        #onehot_labels = tf.manip.reshape(onehot_labels, [-1, params["batch_size"]])        

        predictions = tf.cast(logits, tf.int64)
        labels = tf.cast(labels, tf.int64)
        #neg_log_likelihood = tf.reduce_mean(-logits_prob.log_prob(labels))
        neg_log_likelihood = logits_prob.cross_entropy(tfd.Categorical(logits=tf.cast(onehot_labels, tf.float32)))
        neg_log_likelihood = tf.reduce_mean(neg_log_likelihood)


        loss = neg_log_likelihood
        
        #code_predictions = code_posterior.sample()
        accuracy = tf.reduce_mean(tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int64)))
        tf.summary.scalar("classification accuracy", accuracy)


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
                "classification_loss": tf.metrics.mean(loss),
                "accuracy": tf.metrics.mean(accuracy),
                "neg_log_likelihood": tf.metrics.mean(neg_log_likelihood),
                "kl": tf.metrics.mean(kl)
            },
        )
