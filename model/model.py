from math import sqrt

import tensorflow as tf
tf.random.set_random_seed(1950)


weight_init = tf.contrib.layers.xavier_initializer()
weight_regularizer = tf.contrib.layers.l2_regularizer(0.001)


def master_module(layer_shape, blocks, kernels, flops_multiplier=None, depth=None, width=None, inference=False):
    if flops_multiplier:
        depth = 1
        width = sqrt(flops_multiplier / depth)

    d = depth
    ch = int(width * 8)

    if not inference:
        training = tf.placeholder(tf.bool, shape=[], name="is_training")
    else:
        training = False

    pose_input = tf.placeholder(dtype=tf.float32, shape=layer_shape, name="pose_input")
    x = pose_input
    x = conv2d(x, channels=8, kernel=1, stride=1, scope='ReduceConv')
    x = conv2d(x, channels=ch, kernel=1, stride=1, scope='ReduceConv2')

    for i in range(d):
        x = blocks[0](x, channels=ch, kernel=kernels[0], train=training, scope=f'Block_0_{i}')

    x = blocks[1](x, channels=ch, kernel=kernels[1], train=training, scope='BlockUpsample_0')
    x = max_pooling(x)

    for i in range(d):
        x = blocks[2](x, channels=ch * 2, kernel=kernels[2], train=training, scope=f'Block_1_{i}')
    for i in range(d):
        x = blocks[3](x, channels=ch * 2, kernel=kernels[3], train=training, scope=f'Block_2_{i}')

    x = blocks[4](x, channels=ch * 12, kernel=kernels[4], train=training, scope='BlockUpsample_1')
    x = max_pooling(x)

    for i in range(d):
        x = blocks[5](x, channels=ch * 24, kernel=kernels[5], train=training, scope=f'Block_3_{i}')
    x = max_pooling(x)

    x = fully_connected(x, units=256, scope='Hidden')  # 32
    y = fully_connected(x, units=1, scope='PoseOutput_y')
    p = fully_connected(x, units=1, scope='PoseOutput_p')
    r = fully_connected(x, units=1, scope='PoseOutput_r')
    output = concat([y, p, r], axis=1, name="pose_output")
    return pose_input, output, training


def ResBlock(x_init, channels, kernel=3, use_bias=True, train=True, scope='resblock'):
    with tf.variable_scope(scope):
        x = conv2d(x_init, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, is_training=train, scope='batch_norm_0')
        x = parametric_relu(x, name='alpha_0')

        x = conv2d(x, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_1')
        x = batch_norm(x, is_training=train, scope='batch_norm_1')
        return x + x_init


def ResBlockUpscale(x_init, channels, kernel=3, use_bias=True, train=True, scope='resblock_reduction'):
    with tf.variable_scope(scope):
        x = conv2d(x_init, channels * 2, kernel=kernel, stride=2, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, is_training=train, scope='batch_norm_0')
        x = parametric_relu(x, name='alpha_0')

        x = conv2d(x, channels * 2, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_1')
        x = batch_norm(x, is_training=train, scope='batch_norm_1')
        x = parametric_relu(x, name='alpha_1')
        return x


def ConvBlock(x_init, channels, kernel=1, use_bias=True, train=True, scope='identity'):
    with tf.variable_scope(scope):
        x = conv2d(x_init, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_0')
        return x


def ConvBlockUpscale(x_init, channels, kernel=1, use_bias=True, train=True, scope='identity_reduction'):
    with tf.variable_scope(scope):
        x1 = conv2d(x_init, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_0')
        x2 = conv2d(x_init, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_1')
        x = concat([x1, x2])
        x = batch_norm(x, is_training=train, scope='batch_norm_0')
        x = parametric_relu(x, name='alpha_0')
        return x


def InvertedResBlock(x_init, channels, kernel=1, use_bias=True, train=True, scope='inv_resblock'):
    with tf.variable_scope(scope):
        x = conv2d(x_init, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, is_training=train, scope="batch_norm_0")
        x = parametric_relu(x, "alpha_0")

        x = dw_conv2d(x, channels, kernel=1, stride=1)
        x = batch_norm(x, is_training=train, scope="batch_norm_1")
        x = parametric_relu(x, "alpha_1")

        x = conv2d(x, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_1')
        x = batch_norm(x, is_training=train, scope="batch_norm_2")
        return x + x_init


def InvertedResBlockRUpscale(x_init, channels, kernel=1, use_bias=True, train=True, scope='inv_resblock_reduction'):
    with tf.variable_scope(scope):
        x = conv2d(x_init, channels, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, is_training=train, scope="batch_norm_0")
        x = parametric_relu(x, "alpha_0")

        x = dw_conv2d(x, channels, kernel=1, stride=2)
        x = batch_norm(x, is_training=train, scope="batch_norm_1")
        x = parametric_relu(x, "alpha_1")

        x = conv2d(x, channels * 2, kernel=kernel, stride=1, use_bias=use_bias, scope='conv_1')
        x = batch_norm(x, is_training=train, scope="batch_norm_2")
        return x


def DepthWiseSeparateBlock(x_init, channels, kernel=3, use_bias=True, train=True, scope='inv_resblock_reduction'):
    with tf.variable_scope(scope):
        x = dw_conv2d(x_init, channels, kernel=kernel, stride=1, scope="dw_conv_0")
        x = conv2d(x, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, is_training=train, scope="batch_norm_0")
        x = parametric_relu(x, "alpha_0")

        x = dw_conv2d(x, channels, kernel=kernel, stride=1, scope="dw_conv_1")
        x = conv2d(x, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1')
        x = batch_norm(x, is_training=train, scope="batch_norm_1")

        x = x + x_init
        return parametric_relu(x)


def DepthWiseSeparateBlockUpscale(x_init, channels, kernel=3, use_bias=True, train=True,
                                  scope='inv_resblock_reduction'):
    with tf.variable_scope(scope):
        x = dw_conv2d(x_init, channels, kernel=kernel, stride=2, scope="dw_conv_0")
        x = conv2d(x, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, is_training=train, scope="batch_norm_0")
        x = parametric_relu(x, "alpha_0")

        x = dw_conv2d(x, channels, kernel=kernel, stride=1, scope="dw_conv_1")
        x = conv2d(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, scope='conv_1')
        x = batch_norm(x, is_training=train, scope="batch_norm_1")

        return parametric_relu(x)


def conv2d(x, channels, kernel=4, stride=2, padding='SAME', use_bias=False, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel,
                             # kernel_initializer=weight_init,
                             # kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x


def dw_conv2d(x, channels, kernel=3, stride=2, padding='SAME', use_bias=True, scope='dw_conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.separable_conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                       strides=stride, depth_multiplier=6, use_bias=use_bias,
                                       padding=padding)

        return x


def fully_connected(x, units, use_bias=True, use_reg=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        if use_reg:
            x = tf.layers.dense(x, units=units,
                                kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer,
                                use_bias=use_bias)
        else:
            x = tf.layers.dense(x, units=units,
                                use_bias=use_bias)

        return x


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x, is_training=is_training,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True,
                                        fused=True,
                                        scope=scope)


def concat(x, axis=3, name=None):
    return tf.concat(x, axis=axis, name=name)


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap


def avg_pooling(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')


def max_pooling(x):
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')


def relu(x):
    return tf.nn.relu(x)


def parametric_relu(_x, name='Alpha'):
    with tf.variable_scope(name_or_scope=name, default_name="Prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def flatten(x):
    return tf.layers.flatten(x)
