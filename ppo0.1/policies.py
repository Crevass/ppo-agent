import tensorflow as tf
import numpy as np
from distributions import NormalDist

from layers_construct import dense_layer
import time
from Filters import TFRunningMeanStd


#SAVE_PATH = '/home/wang/Research/MODELS/PPO-MODEL/under_coding/'

L1NUM = 64
L2NUM = 64
L3NUM = 64
L4NUM = 64

class MlpPolicy(object):
	def __init__(self, name, ob_space, ac_space, sess):
		with tf.variable_scope(name):
			self.sess = sess
			self._init(ob_space=ob_space, ac_space=ac_space)
			self.scope = tf.get_variable_scope().name
			print(self.scope)
	def _init(self, ob_space, ac_space):
		
		self.ob = tf.placeholder(dtype=tf.float32, shape=[None]+list(ob_space.shape), name='ob')
		# build ob filter
		with tf.variable_scope('ob_filter'):
			self.ob_rms = TFRunningMeanStd(epsilon=1e-2, shape=ob_space.shape, sess=self.sess)
		# filtering ob
		obz = tf.clip_by_value(((self.ob - self.ob_rms.mean) / self.ob_rms.std), -5.0, 5.0)
		# build up policy
		self.pi = self._build_policy(ac_space, ob_space, obz, True, 'pol')
		#build old policy
		self.oldpi = self._build_policy(ac_space, ob_space, obz, False, 'old_pol')
		# build value function
		self.vf = self._build_vf(ac_space, ob_space, obz, True, 'vf')
		
		self.sample_action = tf.squeeze(self.pi.sample(), axis=0)
		self.obtain_v = tf.squeeze(self.vf, axis=0)
###################################### network construction ###########################################################

	def _build_policy(self, ac_space, ob_space, observation, trainable, scope, scale=0.1):
		with tf.variable_scope(scope):
			l1 = dense_layer(observation, L1NUM, tf.tanh, True, trainable, 'fc1', (True and trainable), False, False)
			l2 = dense_layer(l1, L2NUM, tf.tanh, True, trainable, 'fc2', (True and trainable), False, False)
			dist = NormalDist(l2, ac_space, trainable)
		return dist

	def _build_vf(self, ac_space, ob_space, observation, trainable, scope, scale=0.1):
		with tf.variable_scope(scope):
			l1 = dense_layer(observation, L1NUM, tf.tanh, True, trainable, 'fc1', True, False, False)
			l2 = dense_layer(l1, L2NUM, tf.tanh, True, trainable, 'fc2', True, False, False)
			final = dense_layer(l2, 1, None, False, trainable, 'final', True, False, False)
		return final

############################################################################################################
	def step(self, s):
		if s.ndim < 2:
			s = s[np.newaxis, :]
		a, value = self.sess.run([self.sample_action, self.obtain_v] , feed_dict={self.ob: s})
		assert np.isnan(a).sum() == 0, 'output NaN action in this step'
		assert np.isnan(value).sum() == 0, 'output NaN value in this step'
		return a, value

	def get_v(self, s):
		if s.ndim < 2:
			s = s[np.newaxis, :]
		value = self.sess.run(self.obtain_v, feed_dict={self.ob: s})
		assert np.isnan(value).sum() == 0, 'output NaN value in this step'
		return value

	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
	def get_pol_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/pol')
	def get_oldpol_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/old_pol')