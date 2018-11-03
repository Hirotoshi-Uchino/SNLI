
import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.constant([1, 2, 3, 4], name='a')
b = tf.constant([1,1,1,1], name='b')

c = a*b

print(c.eval())