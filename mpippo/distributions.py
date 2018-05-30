import numpy as np
import tensorflow as tf
from common_util import variable_summaries
from layers_construct import dense_layer

class NormalDist(object):
	def __init__(self, x, ac_space, trainable):
		self.logstd = tf.get_variable(
			name='logstd',
			shape=[1]+list(ac_space.shape),
			initializer=tf.zeros_initializer(),
			trainable=trainable
			)
		self.mean = dense_layer(x, ac_space.shape[0], None, True, trainable, 'mean', (True and trainable), False, False)
		self.std = tf.exp(self.logstd)

	def get_mean(self):
		return self.mean

		
	def neglogp(self, x):
		return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1, keepdims=True) \
			+ 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
			+ tf.reduce_sum(self.logstd, axis=-1)
	
	def kl(self, other):
		assert isinstance(other, NormalDist)
		return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
	def entropy(self):
		return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
	def sample(self):
		return self.mean + self.std * tf.random_normal(tf.shape(self.mean))