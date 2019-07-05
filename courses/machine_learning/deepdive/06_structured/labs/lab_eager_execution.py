
#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

#%%
tf.enable_eager_execution()
tf.executing_eagerly()


#%%
x = [[2.]]
m = tf.matmul(x,x)
print("hello, {}".format(m))

#%%
a=tf.constant([[2,3],[3,4]])
print(a)

#%%
b = tf.add(a,1)
print(b)


#%%
