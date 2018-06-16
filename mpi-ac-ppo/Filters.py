import numpy as np
import tensorflow as tf
import mpi4py.MPI as MPI

class RuningMeanStd(object):
	def __init__(self, epsilon = 1e-4, shape = ()):
		self.mean = np.zeros(shape, 'float32')
		self.var = np.ones(shape, 'float32')
		self.count = epsilon

	def update(self, x):
		if x.ndim < 2:
			x = x[np.newaxis, :]
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		delta = batch_mean -self.mean
		total_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / total_count
		m_a = self.var * (self.count)
		m_b = batch_var * (batch_count)
		M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
		new_var = M2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count

	def filtting(self, x):
		return (x - self.mean) / (np.sqrt(self.var))

class MPIRunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	def __init__(self, *, epsilon=1e-2, shape=(), sess):
		self.sess = sess
		self._sum = tf.get_variable(
			dtype=tf.float64,
			shape=shape,
			initializer=tf.constant_initializer(0.0),
			name="runningsum", trainable=False)
		self._sumsq = tf.get_variable(
			dtype=tf.float64,
			shape=shape,
			initializer=tf.constant_initializer(epsilon),
			name="runningsumsq", trainable=False)
		self._count = tf.get_variable(
			dtype=tf.float64,
			shape=(),
			initializer=tf.constant_initializer(epsilon),
			name="count", trainable=False)
		self.shape = shape

		self.mean = tf.to_float(self._sum / self._count)
		self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean) , 1e-2 ))

		self.newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
		self.newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
		self.newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
		"""
		self.incfiltparams = U.function([newsum, newsumsq, newcount], [],
			updates=[
			tf.assign_add(self._sum, newsum),
			tf.assign_add(self._sumsq, newsumsq),
			tf.assign_add(self._count, newcount)]
			)
		"""
		self.update_data = [
		tf.assign_add(self._sum, self.newsum),
		tf.assign_add(self._sumsq, self.newsumsq),
		tf.assign_add(self._count, self.newcount)
		]

	def update(self, x):
		x = x.astype('float64')
		n = int(np.prod(self.shape))
		totalvec = np.zeros(n*2+1, 'float64')
		addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
		MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
		#self.incfiltparams(totalvec[0:n].reshape(self.shape), totalvec[n:2*n].reshape(self.shape), totalvec[2*n])
		self.sess.run(self.update_data, feed_dict={
											self.newsum: totalvec[0:n].reshape(self.shape),
											self.newsumsq: totalvec[n:2*n].reshape(self.shape),
											self.newcount: totalvec[2*n]}
											)