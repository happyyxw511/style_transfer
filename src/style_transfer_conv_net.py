import tensorflow as tf
import numpy as np

WEIGHTS_INIT_STDEV = .1


def style_transfer_conv_net(input):
    pass


def _conv_layer(pre_net, num_filters, filter_size, strides, relu=True):
    num_in_channels = pre_net.get_shape()[3].value
    inited_filter = _conv_init_filter(num_in_channels, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(pre_net, inited_filter, strides, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net


def _residual_block(pre_net, num_filters=128, filter_size=3):
    print pre_net.get_shape()
    conv_layer = _conv_layer(pre_net, num_filters, filter_size, 1)
    print conv_layer.get_shape()
    return pre_net + _conv_layer(conv_layer, num_filters, filter_size, 1, relu=False)


def _instance_norm(net):
    channels = net.get_shape()[3].value
    var_shape = [channels]
    mu, sigma = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    eps = 1e-3
    normalized = (net - mu)/(sigma + eps)**(.5)
    return scale*normalized + shift


def _conv_init_filter(num_in_channels, num_out_channels, filter_size, transpose=False):
    if not transpose:
        weights_shape = [filter_size, filter_size, num_in_channels, num_out_channels]
    else:
        weights_shape = [filter_size, filter_size, num_out_channels, num_in_channels]
    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1),
                               dtype=tf.float32)
    return weights_init
