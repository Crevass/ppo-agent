import tensorflow as tf
import numpy as np
import random


def variable_summaries(var, scope):
	with tf.name_scope(scope):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		#with tf.name_scope('stddev'):
		#	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		#tf.summary.scalar('stddev', stddev)
		#tf.summary.scalar('max', tf.reduce_max(var))
		#tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


def weight_variable(input_dim, output_dim, trainable, scale=0.1):
	return tf.get_variable(
		name='weights',
		shape=[input_dim, output_dim],
		initializer=tf.truncated_normal_initializer(stddev=0.1),
		trainable=trainable)

def bias_variable(output_dim, trainable):
	return tf.get_variable(
		name='bias',
		shape=[1,output_dim],
		initializer=tf.constant_initializer(0.1),
		trainable=trainable)


class Memory(object):
	def __init__(self):
		self.actions = None
		self.observation = None
		self.adv = None
		self.v = None
		self.target_v = None
		self.size = 0
	def save(self, s, a, advs, target_v, v):
		if (self.size == 0):
			self.observation = s.copy()
			self.actions = a.copy()
			self.adv = advs.copy()
			self.target_v = target_v.copy()
			self.v = v.copy()
			self.size = self.observation.shape[0]
		else:	
			self.observation = np.concatenate([self.observation, s.copy()], axis=0)
			self.actions = np.concatenate([self.actions, a.copy()], axis=0)
			self.adv = np.concatenate([self.adv, advs.copy()], axis=0)
			self.target_v = np.concatenate([self.target_v, target_v.copy()], axis=0)
			self.v = np.concatenate([self.v, v.copy()], axis=0)
			self.size = self.observation.shape[0]

	def sample(self, *, batch_size):
		assert self.size >= batch_size, 'Not enough transitions'
		indices = random.sample(range(self.size), batch_size)
		bs = self.observation[indices, :].copy()
		ba = self.actions[indices, :].copy()
		badv = self.adv[indices, :].copy()
		btarv = self.target_v[indices, :].copy()
		boldv = self.v[indices, :].copy()
		return bs, ba, badv, btarv, boldv

	def sample_all(self):
		return self.observation.copy(), self.actions.copy(), self.adv.copy(), self.target_v.copy(), self.v.copy()

	def reset(self):
		self.actions = None
		self.observation = None
		self.adv = None
		self.target_v = None
		self.v = None
		self.size = 0

	def get_size(self):
		if isinstance(self.observation, np.ndarray):
			assert self.size == self.observation.shape[0], 'replay inside error'
		else:
			assert self.size == 0
		return self.size