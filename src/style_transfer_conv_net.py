import tensorflow as tf

WEIGHTS_INIT_STDEV = .1


def net(input):
    conv1 = _conv_layer(input, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3)
    resid2 = _residual_block(resid1)
    resid3 = _residual_block(resid2)
    resid4 = _residual_block(resid3)
    resid5 = _residual_block(resid4)
    conv_t1 = _conv_transpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_transpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds


def _conv_layer(pre_net, num_filters, filter_size, strides, relu=True):
    num_in_channels = pre_net.get_shape()[3].value
    inited_filter = _conv_init_filter(num_in_channels, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(pre_net, inited_filter, strides_shape, padding='SAME')
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


def _conv_transpose_layer(pre_net, num_filters, filter_size, strides):
    num_in_channels = pre_net.get_shape()[3].value
    inited_filter = _conv_init_filter(num_in_channels, num_filters, filter_size, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in pre_net.get_shape()]
    new_rows, new_cols = int(rows*strides), int(cols*strides)
    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(pre_net, inited_filter, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)

