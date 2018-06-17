import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from distributions import NormalDist
import mpi4py.MPI as MPI
from layers_construct import dense_layer, fc, res_connection
import time
#from Filters import MPIRunningMeanStd


#SAVE_PATH = '/home/wang/Research/MODELS/PPO-MODEL/under_coding/'

L1NUM = 64
L2NUM = 64
L3NUM = 128
L4NUM = 128

class MlpPolicy(object):
	def __init__(self, name, ob_space, ac_space, sess):
		with tf.variable_scope(name):
			self.sess = sess
			self._init(ob_space=ob_space, ac_space=ac_space)
			self.scope = tf.get_variable_scope().name
			print(self.scope + "-" + str(MPI.COMM_WORLD.Get_rank()) + " initialized")
	def _init(self, ob_space, ac_space):
		
		self.ob = tf.placeholder(dtype=tf.float32, shape=[None]+list(ob_space.shape), name='ob')
		#self.ob_ = tf.placeholder(dtype=tf.float32, shape=[None]+list(ob_space.shape), name='ob_')
		# build ob filter
		#with tf.variable_scope('ob_filter'):
		#	self.ob_rms = MPIRunningMeanStd(epsilon=1e-2, shape=ob_space.shape, sess=self.sess)
		
		# filtering ob
		#obz = tf.clip_by_value(((self.ob - self.ob_rms.mean) / self.ob_rms.std), -5.0, 5.0)
		#obz = ((self.ob - self.ob_rms.mean) / self.ob_rms.std)
		obz = self.ob
		#obz_ = self.ob_
		
		# build up policy
		self.pi = self._build_policy(ac_space, ob_space, obz, True, 'pol')
		#build old policy
		self.oldpi = self._build_policy(ac_space, ob_space, obz, False, 'old_pol')
		
		# build value function
		self.vf = self._build_vf(ac_space, ob_space, obz, True, 'vf')
			
		# build critic
		self.critic = self._build_critic(ac_space, ob_space, self.pi, obz, True, 'critic')
		# build old critic
		#self.oldcritic = self._build_critic(ac_space, ob_space, self.oldpi.mean, obz_, False, 'old_critic')

		self.sample_action = tf.squeeze(self.pi.sample(), axis=0)
		self.obtain_v = tf.squeeze(self.vf, axis=0)
###################################### network construction ###########################################################

	def _build_policy(self, ac_space, ob_space, ob, trainable, scope, scale=0.1):
		with tf.variable_scope(scope):
			l1 = tf.tanh(fc(ob, L1NUM, True, trainable, 'fc1', (True and trainable), (True and trainable), False))
			l2 = tf.tanh(fc(l1, L2NUM, True, trainable, 'fc2', (True and trainable), (True and trainable), False))
			#l3 = tf.tanh(fc(l2, L3NUM, True, trainable, 'fc3', (True and trainable), (True and trainable), False))
			#res_l3 = tf.tanh(res_connection(l1, l3, trainable, 'br1'))
			#l4 = fc(res_l3, L4NUM, True, trainable, 'fc4', (True and trainable), False, False)
			#res_l4 = tf.tanh(res_connection(l1, l4, trainable, 'br2'))
			dist = NormalDist(l2, ac_space, trainable)
		return dist

	def _build_vf(self, ac_space, ob_space, ob, trainable, scope, scale=0.1):
		with tf.variable_scope(scope):
			l1 = dense_layer(ob, L1NUM, tf.tanh, True, trainable, 'fc1', True, True, False)
			l2 = dense_layer(l1, L2NUM, tf.tanh, True, trainable, 'fc2', True, True, False)
			#l3 = dense_layer(l2, L3NUM, tf.tanh, True, trainable, 'fc3', True, False, False)
			final = dense_layer(l2, 1, None, False, trainable, 'final', True, True, False)
		return final

	def _build_critic(self, ac_space, ob_space, ac, ob, trainable, scope, layer_norm=True):
		with tf.variable_scope(scope):
			x = fc(ob, L1NUM, True, trainable, 'fc1_s')
			if layer_norm:
				x = tc.layers.layer_norm(x, center=True, scale=True)
			x = tf.nn.relu(x)

			x = tf.concat([x, ac], axis=-1)

			x = fc(x, L2NUM, True, trainable, 'fc2')
			if layer_norm:
				x = tc.layers.layer_norm(x, center=True, scale=True)
			x = tf.nn.relu(x)

			final = fc(x, 1, True, trainable, 'final')
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
	def get_critic_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/critic')
	#def get_oldcritic_variables(self):
	#	return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/oldcritic')
	def get_ppo_variables(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/pol') + 
				tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/old_pol') +
				tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/vf')