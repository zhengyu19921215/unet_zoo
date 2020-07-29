import tensorflow as tf 
import numpy as np
import tensorflow.contrib.slim as slim
import os
reg=slim.l2_regularizer(scale=0.001)

def conv_block(inputs,stage,nb_filter,is_training,kernel_size=3):
    x=slim.conv2d(inputs,num_outputs=nb_filter,kernel_size=kernel_size,stride=1,rate=1,activation_fn=None,weigt_regularizer=reg)
    x=slim.batch_norm(x,is_training=is_training)
    x=tf.nn.leaky_relu(x)
    x=slim.conv2d(x,num_outputs=nb_filter,kernel_size=kernel_size,stride=1,rate=1,activation_fn=None,weigt_regularizer=reg)
    x=slim.batch_norm(x,is_training=is_training)
    x=tf.nn.leaky_relu(x)
    weight = x.get_shape()[1].value
    height = x.get_shape()[2].value
def img_scale(x, scale):
    weight = x.get_shape()[1].value
    height = x.get_shape()[2].value

    try:
        out = tf.image.resize_nearest_neighbor(x, size=(weight*scale, height*scale))
    except:
        out = tf.image.resize_images(x, size=[weight*scale, height*scale])
    return out
def FSK_unit(x,nb_filter,is_training,max_pooling=0,upsampling=0):
   
    _,_,_,C=x.get_shape().as_list()
    if max_pooling:
        x=slim.max_pool2d(x,[max_pooling,max_pooling],max_pooling)
    elif upsampling:
        x=img_scale(x,upsampling)
       
    x=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None)
    x=slim.batch_norm(x,is_training=is_training)
    x=tf.nn.leaky_relu(x)     
    return x
 def UNet_ppp(inputs,is_training,min_filter=16,deep_supervision=False,CGM=False):
     
