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

def fc(input_layer, nn_num, use_bias, trainable, scope, monitor_weight=False, monitor_bias=False, monitor_output=False):
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
		if monitor_output:
			variable_summaries(preactive, 'preactive')
	return preactive


def res_connection(inpt, oupt, trainable, scope):
	if inpt.shape == oupt.shape:
		x = tf.add(inpt, oupt)
	else:
		branch = fc(inpt, oupt.shape[-1], True, trainable, scope)
		x = tf.add(branch, oupt)
	return x