from utils import *
from layers.pooling import max_pool_2d
import tensorflow as tf


def conv2d_transpose(name, x, w=None, output_shape=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     l2_strength=0.0,
                     bias=0.0, activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                     is_training=True):
    """
    This block is responsible for a convolution transpose 2D followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param output_shape: (Array) [N, H', W', C'] The shape of the corresponding output.
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = conv2d_transpose_p(name=scope, x=x, w=w, output_shape=output_shape, kernel_size=kernel_size,
                                      padding=padding, stride=stride,
                                      l2_strength=l2_strength,
                                      bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr

        return conv_o


def conv2d_transpose_p(name, x, w=None, output_shape=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                       l2_strength=0.0,
                       bias=0.0):
    """
    Convolution Transpose 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param output_shape: (Array) [N, H', W', C'] The shape of the corresponding output.
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.shape.as_list()[-1]]
        if w == None:
            w = get_deconv_filter(kernel_shape, l2_strength)
        variable_summaries(w)
        deconv = tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=stride, padding=padding)
        if isinstance(bias, float):
            bias = tf.get_variable('layer_biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        variable_summaries(bias)
        out = tf.nn.bias_add(deconv, bias)

    return out
