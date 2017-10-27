#! /usr/bin/python
# -*- coding: utf8 -*-
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from config import *


# define binary layer and calculation
class BinaryConv2dLayer(Layer):
    """
    The :class:`BinaryConv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [filter_height, filter_width, in_channels, out_channels].
    strides : a list of ints.
        The stride of the sliding window for each dimension of input.\n
        It Must be in the same order as the dimension specified with format.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    use_cudnn_on_gpu : an optional string from: "NHWC", "NCHW". Defaults to "NHWC".
    data_format : an optional bool. Defaults to True.
    name : a string or None
        An optional name to attach to this layer.

    Notes
    ------
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.BinaryConv2dLayer(network,
    ...                   act = tf.nn.relu,
    ...                   shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    ...                   strides=[1, 1, 1, 1],
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   W_init_args={},
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> network = tl.layers.PoolLayer(network,
    ...                   ksize=[1, 2, 2, 1],
    ...                   strides=[1, 2, 2, 1],
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)

    >>> Without TensorLayer, you can implement 2d convolution as follow.
    >>> W = tf.Variable(W_init(shape=[5, 5, 1, 32], ), name='W_conv')
    >>> outputs = tf.nn.relu( tf.nn.conv2d(inputs, W,
    ...                       strides=[1, 1, 1, 1],
    ...                       padding='SAME') )
    """

    def __init__(self,
                 layer=None,
                 act=tf.identity,
                 shape=[5, 5, 1, 100],
                 strides=[1, 1, 1, 1],
                 padding='SAME',
                 W_init=tf.truncated_normal_initializer(stddev=0.02),
                 W_init_args={},
                 use_cudnn_on_gpu=None,
                 data_format=None,
                 name='cnn_layer',
                 is_train=False,
                 binary=False):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print(
            "  [TL] BinaryConv2dLayer %s: shape:%s strides:%s pad:%s act:%s" %
            (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            if binary:
                if is_train:
                    # Enable binarization, different naming category
                    W = tf.get_variable(
                        name='W_conv2d_real',
                        shape=shape,
                        initializer=W_init,
                        **W_init_args)
                    W_b = tf.get_variable(
                        name='W_conv2d_binary',
                        shape=shape,
                        initializer=tf.constant_initializer(value=0.0),
                        **W_init_args)
                    # binarization of W to W_b
                    W_b = tf.sign(W) * (
                        tf.reduce_sum(W) / tf.reduce_sum(tf.sign(W)))
                    self.outputs = act(
                        tf.nn.conv2d(
                            self.inputs,
                            W_b,
                            strides=strides,
                            padding=padding,
                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                            data_format=data_format))
                else:
                    # directly infer from W_b
                    W_b = tf.get_variable(
                        name='W_conv2d_binary',
                        shape=shape,
                        initializer=tf.constant_initializer(value=0.0),
                        **W_init_args)
                    self.outputs = act(
                        tf.nn.conv2d(
                            self.inputs,
                            W_b,
                            strides=strides,
                            padding=padding,
                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                            data_format=data_format))
            else:
                W = tf.get_variable(
                    name='W_conv2d',
                    shape=shape,
                    initializer=W_init,
                    **W_init_args)
                self.outputs = act(
                    tf.nn.conv2d(
                        self.inputs,
                        W,
                        strides=strides,
                        padding=padding,
                        use_cudnn_on_gpu=use_cudnn_on_gpu,
                        data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if binary:
            self.all_params.extend([W_b])
            if is_train:
                self.all_params.extend([W])
        else:
            self.all_params.extend([W])


def get_variables_with_name_in_binary_training(name,
                                               train_only=True,
                                               printable=False):
    """Get variable list by a given name scope.

    Examples
    ---------
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()

    d_vars = []
    for var in t_vars:
        if name in var.name and 'real' not in var.name:
            d_vars.append(var)
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name,
                                                  str(v.get_shape())))
    return d_vars


def process_grads(grads_and_vars, binary=False):
    """ Customized gradients processing function: apply the binary weights gradients to the original real weights
    sandwiched by compute_gradients() and apply_gradients()
    """
    if binary:
        # apply the binary weights gradients to the original real weights
        new_grads_and_vars = []
        for grad_and_var in grads_and_vars:
            if 'binary' in grad_and_var[1].name:
                binary_naming = grad_and_var[1].name
                real_naming = binary_naming.replace('binary', 'real')
                real_var = tf.get_variable(name=real_naming)
                new_grads_and_vars.append((grad_and_var[0], real_var))
            else:
                new_grads_and_vars.append(grad_and_var)
        return new_grads_and_vars
    return grads_and_vars


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def LapSRNSingleLevel(net_image, net_feature, reuse=False, binary=False):
    with tf.variable_scope("Model_level", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_tmp = net_feature
        # recursive block
        for d in range(config.model.resblock_depth):
            net_tmp = PReluLayer(net_tmp, name='prelu_D%s' % (d))
            net_tmp = BinaryConv2dLayer(
                net_tmp,
                shape=[3, 3, 64, 64],
                strides=[1, 1, 1, 1],
                name='conv_D%s' % (d),
                W_init=tf.contrib.layers.xavier_initializer(),
                binary=binary)
            # recursive_scope.reuse_variables()
            # for r in range(1,config.model.recursive_depth):
            #     # recursive block
            #     for d in range(config.model.resblock_depth):
            #         net_tmp = Conv2dLayer(net_tmp,shape=[3,3,64,64],strides=[1,1,1,1],
            #                             act=lrelu,name='Level%s_D%s_conv'%(level,d))
        net_feature = ElementwiseLayer(
            layer=[net_feature, net_tmp],
            combine_fn=tf.add,
            name='add_feature')

        net_feature = PReluLayer(net_feature, name='prelu_feature')
        net_feature = BinaryConv2dLayer(
            net_feature,
            shape=[3, 3, 64, 256],
            strides=[1, 1, 1, 1],
            name='upconv_feature',
            W_init=tf.contrib.layers.xavier_initializer(),
            binary=binary)
        net_feature = SubpixelConv2d(
            net_feature, scale=2, n_out_channel=64, name='subpixel_feature')

        # add image back
        gradient_level = BinaryConv2dLayer(
            net_feature,
            shape=[3, 3, 64, 3],
            strides=[1, 1, 1, 1],
            name='grad',
            W_init=tf.contrib.layers.xavier_initializer(),
            binary=binary)
        net_image = Conv2dLayer(
            net_image,
            shape=[3, 3, 3, 12],
            strides=[1, 1, 1, 1],
            name='upconv_image',
            W_init=tf.contrib.layers.xavier_initializer())
        net_image = SubpixelConv2d(
            net_image, scale=2, n_out_channel=3, name='subpixel_image')
        net_image = ElementwiseLayer(
            layer=[gradient_level, net_image],
            combine_fn=tf.add,
            name='add_image')

    return net_image, net_feature, gradient_level


def LapSRN(inputs, is_train=False, reuse=False, binary=False):
    n_level = int(np.log2(config.model.scale))
    assert n_level >= 1

    with tf.variable_scope("LapSRN", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        shapes = tf.shape(inputs)
        inputs_level = InputLayer(inputs, name='input_level')

        net_feature = Conv2dLayer(
            inputs_level,
            shape=[3, 3, 3, 64],
            strides=[1, 1, 1, 1],
            W_init=tf.contrib.layers.xavier_initializer(),
            name='init_conv')
        net_image = inputs_level

        # net_image, net_feature, net_gradient = LapSRNSingleLevel(net_image, net_feature, reuse=reuse)
        # for level in range(1,n_level):
        # net_image, net_feature, net_gradient = LapSRNSingleLevel(net_image, net_feature, reuse=True)

        net_image1, net_feature1, net_gradient1 = LapSRNSingleLevel(
            net_image, net_feature, reuse=reuse, binary=binary)

        net_image2, net_feature2, net_gradient2 = LapSRNSingleLevel(
            net_image1, net_feature1, reuse=True, binary=binary)
        # For 8x, we just add another layer

    return net_image2, net_gradient2, net_image1, net_gradient1     # both 2x and 4x features
