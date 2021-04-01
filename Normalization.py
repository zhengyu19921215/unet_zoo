import tensorflow as tf


#BN
def BN(input, bn_training, name):
    # assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool
    def bn(input, bn_training, name, reuse=None):
        with tf.variable_scope('bn', reuse=reuse):
            output = tf.layers.batch_normalization(input, training=bn_training, name=name)
        return output
    return tf.cond(
        bn_training,
        lambda: bn(input, bn_training=True, name=name, reuse=None),
        lambda: bn(input, bn_training=False, name=name, reuse=True),
    )
#LN
def Layernorm(x, gamma, beta):
    # x_shape:[B, H, W, C]
    results = 0.
    eps = 1e-5
    x = tf.transpose(x, [0, 3, 1, 2])  # [B,C,H,W]
    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    results = tf.transpose(results, [0, 2, 3, 1])
    return results

#IN
def Instancenorm(x, gamma, beta):
    # x_shape:[B, H, W, C]
    results = 0.
    eps = 1e-5
    x = tf.transpose(x, [0, 3, 1, 2]) #[B,C,H,W]
    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.var(x, axis=(2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    #
    results = tf.transpose(results, [0, 2, 3, 1])
    return results
#GN
def GN(x, traing_flag, scope, G=16, esp=1e-5):
    # tranpose:[bs,h,w,c]to[bs,c,h,w]follwing the paper
    x = tf.transpose(x, [0, 3, 1, 2])
    N, C, H, W = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [-1, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gama and beta
    gama = tf.get_variable(scope + 'group_gama', [C], initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable(scope  + 'group_beta', [C], initializer=tf.constant_initializer(0.0))
    gama = tf.reshape(gama, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])
    output = tf.reshape(x, [-1, C, H, W]) * gama + beta
    ## tranpose:[bs,c,h,w]to[bs,h,w,c]follwing the paper
    output = tf.transpose(output, [0, 2, 3, 1])
    return output

#SN
def SwitchableNorm(x, gamma, beta, w_mean, w_var):
    # x_shape:[B, H, W, C]
    results = 0.
    eps = 1e-5
    x = tf.transpose(x, [0, 3, 1, 2])  # [B,C,H,W]
    mean_in = np.mean(x, axis=(2, 3), keepdims=True)
    var_in = np.var(x, axis=(2, 3), keepdims=True)

    mean_ln = np.mean(x, axis=(1, 2, 3), keepdims=True)
    var_ln = np.var(x, axis=(1, 2, 3), keepdims=True)

    mean_bn = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var_bn = np.var(x, axis=(0, 2, 3), keepdims=True)

    mean = w_mean[0] * mean_in + w_mean[1] * mean_ln + w_mean[2] * mean_bn
    var = w_var[0] * var_in + w_var[1] * var_ln + w_var[2] * var_bn

    x_normalized = (x - mean) / np.sqrt(var + eps)
    results = gamma * x_normalized + beta
    results = tf.transpose(results, [0, 2, 3, 1])
    return results
