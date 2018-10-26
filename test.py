import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


tfd = tfp.distributions

codes = tf.one_hot([2 for i in range(16)], 10) 

codes = tf.reshape(codes, (-1, 1, 1, 10))


sess = tf.Session()
with sess.as_default():
    print(codes.eval())
"""
p = [[0.6, 0.2,0.5,1.2],[0.01, 0.02 ,12, 0.01]]
ohc = tfd.Categorical(logits=p)
s = ohc.sample(16)

label = [[1.,0,0,0],[0,0,1.,0]]
ohl = tfd.Categorical(logits=label)
a = ohc.cross_entropy(ohl)

sess = tf.Session()
with sess.as_default():
    print(sess.run(tf.reduce_sum(ohc.probs, 1)))
"""


#from tensorflow.contrib.learn.python.learn.datasets import mnist
#mnist_data = mnist.read_data_sets(FLAGS.data_dir, reshape=False)
#print(mnist_data)

