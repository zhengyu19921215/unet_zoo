import tensorflow as tf
import numpy as np


def xavier_initializer_convolution(shape, dist='uniform', lambda_initializer=True):
    """
    Xavier initializer for N-D convolution patches. input_activations = patch_volume * in_channels;
    output_activations = patch_volume * out_channels; Uniform: lim = sqrt(3/(input_activations + output_activations))
    Normal: stddev =  sqrt(6/(input_activations + output_activations))
    :param shape: The shape of the convolution patch i.e. spatial_shape + [input_channels, output_channels]. The order of
    input_channels and output_channels is irrelevant, hence this can be used to initialize deconvolution parameters.
    :param dist: A string either 'uniform' or 'normal' determining the type of distribution
    :param lambda_initializer: Whether to return the initial actual values of the parameters (True) or placeholders that
    are initialized when the session is initiated
    :return: A numpy araray with the initial values for the parameters in the patch
    """
    s = len(shape) - 2
    num_activations = np.prod(shape[:s]) * np.sum(shape[s:])  # input_activations + output_activations
    if dist == 'uniform':
        lim = np.sqrt(6. / num_activations)
        if lambda_initializer:
            return np.random.uniform(-lim, lim, shape).astype(np.float32)
        else:
            return tf.random_uniform(shape, minval=-lim, maxval=lim)
    if dist == 'normal':
        stddev = np.sqrt(3. / num_activations)
        if lambda_initializer:
            return np.random.normal(0, stddev, shape).astype(np.float32)
        else:
            tf.truncated_normal(shape, mean=0, stddev=stddev)
    raise ValueError('Distribution must be either "uniform" or "normal".')


def constant_initializer(value, shape, lambda_initializer=True):
    if lambda_initializer:
        return np.full(shape, value).astype(np.float32)
    else:
        return tf.constant(value, tf.float32, shape)


def get_spatial_rank(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
    """
    return len(x.get_shape()) - 2


def get_num_channels(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.get_shape()[-1])


def get_spatial_size(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: The spatial shape of x, excluding batch_size and num_channels.
    """
    return x.get_shape()[1:-1]


# parametric leaky relu
def prelu(x):
    alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def convolution(x, filter, padding='SAME', strides=None, dilation_rate=None):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-1]))

    return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b


def deconvolution(x, filter, output_shape, strides, padding='SAME'):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-2]))

    spatial_rank = get_spatial_rank(x)
    if spatial_rank == 2:
        return tf.nn.conv2d_transpose(x, filter, output_shape, strides, padding) + b
    if spatial_rank == 3:
        return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
    raise ValueError('Only 2D and 3D images supported.')


# More complex blocks

# down convolution
def down_convolution(x, factor, kernel_size):
    num_channels = get_num_channels(x)
    spatial_rank = get_spatial_rank(x)
    strides = spatial_rank * [factor]
    filter = kernel_size + [num_channels, num_channels * factor]
    x = convolution(x, filter, strides=strides)
    return x


# up convolution
def up_convolution(x, output_shape, factor, kernel_size):
    num_channels = get_num_channels(x)
    spatial_rank = get_spatial_rank(x)
    strides = [1] + spatial_rank * [factor] + [1]
    filter = kernel_size + [num_channels // factor, num_channels]
    x = deconvolution(x, filter, output_shape, strides=strides)
    return x
 def convolution_block(layer_input, num_convolutions, keep_prob, activation_fn, is_training):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution(x, [5, 5, 5, n_channels, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            # layer_input = tf.layers.batch_normalization(layer_input, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, keep_prob, activation_fn, is_training):
    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    n_channels = get_num_channels(layer_input)
    if num_convolutions == 1:
        with tf.variable_scope('conv_' + str(1)):
            x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            # layer_input = tf.layers.batch_normalization(layer_input, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
        return x

    with tf.variable_scope('conv_' + str(1)):
        x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
        x = activation_fn(x)
        x = tf.nn.dropout(x, keep_prob)

    for i in range(1, num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution(x, [5, 5, 5, n_channels, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            # layer_input = tf.layers.batch_normalization(layer_input, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)

    return x

class VNet(object):
    def __init__(self,
                 num_classes,
                 keep_prob=1.0,
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 is_training = True,
                 activation_fn="relu"):
        """
        Implements VNet architecture https://arxiv.org/abs/1606.04797
        :param num_classes: Number of output classes.
        :param keep_prob: Dropout keep probability, set to 0.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation_fn: The activation function.
        """
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.is_training = is_training
        self.train_phase = tf.placeholder(tf.bool,name="train_phase_placeholder")

        if (activation_fn == "relu"):
            self.activation_fn = tf.nn.relu
        elif(activation_fn == "prelu"):
            self.activation_fn = prelu

    def network_fn(self, x):
        # keep_prob = self.keep_prob if self.is_training else 1.0
        # use 0.0 for tf 1.15
        # keep_prob = self.keep_prob if self.is_training else 0.0
        keep_prob = self.keep_prob
        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
        # channel
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope('vnet/input_layer'):
            if input_channels == 1:
                x = tf.tile(x, [1, 1, 1, 1, self.num_channels])
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
            else:
                x = convolution(x, [5, 5, 5, input_channels, self.num_channels])
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
                x = self.activation_fn(x)

        features = list()

        for l in range(self.num_levels):
            with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                x = convolution_block(x, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn, is_training=self.train_phase)
                features.append(x)
                with tf.variable_scope('down_convolution'):
                    x = down_convolution(x, factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
                    x = self.activation_fn(x)

        with tf.variable_scope('vnet/bottom_level'):
            x = convolution_block(x, self.bottom_convolutions, keep_prob, activation_fn=self.activation_fn, is_training=self.train_phase)

        for l in reversed(range(self.num_levels)):
            with tf.variable_scope('vnet/decoder/level_' + str(l + 1)):
                f = features[l]
                with tf.variable_scope('up_convolution'):
                    x = up_convolution(x, tf.shape(f), factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
                    x = self.activation_fn(x)

                x = convolution_block_2(x, f, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn, is_training=self.train_phase)

        with tf.variable_scope('vnet/output_layer'):
            logits = convolution(x, [1, 1, 1, self.num_channels, self.num_classes])
            logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)

        return logits
