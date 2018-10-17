import numpy as np
latent_size = 16
loc_modify = np.array([[i%10-5 if int(i/10)==j else 0 for j in range(latent_size)] for i in range(latent_size*10)])

print(loc_modify)
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