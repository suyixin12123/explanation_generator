import numpy as np
import tensorflow as tf



train, test= tf.keras.datasets.fashion_mnist.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((train[0], train[1]))
train_dataset = train_dataset.batch(32)

train_data = train_dataset.shuffle(1000).repeat().batch(32)
def _parser(s):
    reshaped = tf.reshape(s, [28,28, 1])

print(train_data)

exit()







latent_size = 16
loc_modify = np.array([[(i%10)*0.1-0.5 if int(i/10)==j else 0 for j in range(latent_size)] for i in range(latent_size*10)])
loc_modify = tf.constant(loc_modify)

loc_normal = np.array([[0. for i in range(latent_size)] for j in range(10*latent_size)])
loc_normal = tf.constant(loc_normal)
loc = loc_modify + loc_normal

sess = tf.Session()
with sess.as_default():
    print(loc.eval())
print(loc)
exit()



import numpy as np

import tensorflow_probability as tfp
import tensorflow as tf


tfd = tfp.distributions

mvn = tfd.MultivariateNormalDiag(
    loc = [1., -1],
    scale_diag=[1,2.]
)
new_loc = mvn.loc + [1., -1]
new_var = mvn.variance()
print(new_var)
mvn1 = tfd.MultivariateNormalDiag(
    loc = new_loc,
    scale_diag=new_var
)
#mvn.loc = [2.,-2]

a = tf.constant([[1,2,3,4,5]])

print(a)
at = tf.transpose(a)
print(at)
att = tf.manip.tile(a, [10,1])
print(att)
sess = tf.Session()
with sess.as_default():
    print(att.eval())