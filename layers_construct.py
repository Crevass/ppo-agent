import tensorflow as tf
import numpy as np
from common_util import variable_summaries
from common_util import weight_variable
from common_util import bias_variable

def dense_layer(input_layer, nn_num, act, use_bias, trainable, scope, monitor_weight=False, monitor_bias=False, monitor_output=False):
	size = input_layer.shape[-1]
	with tf.variable_scope(scope, reuse=False):
		w = weight_variable(size, nn_num, trainable)
		if monitor_weight:
			variable_summaries(w, 'weights')
		mul = tf.matmul(input_layer, w)
		if use_bias:
			b = bias_variable(nn_num, trainable)
			if monitor_bias:
				variable_summaries(b, 'bias')
			with tf.name_scope('pre_active'):
				preactive = tf.add(mul, b)
		else:
			with tf.name_scope('pre_active'):
				preactive = mul
		with tf.name_scope('activation'):
			if (act != None):
				activation = act(preactive)
			else:
				activation = preactive
		if monitor_output:
			variable_summaries(activation, 'activated')
	return activation


def conv2d(*, x, num_filters, name, filter_size=(3,3), stride=(1,1), pad="SAME", dtype=tf.float32, collections=None, summary_tag=None):
	with tf.variable_scope(name):
		stride_shape = [1, stride[0], stride[1], 1]
		filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

		fan_in = intprod(filter_shape[:3])

		fan_out = intprob