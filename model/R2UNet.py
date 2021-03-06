import tensorflow as tf 
import numpy as np
import tensorflow.contrib.slim as slim
import os
reg=slim.l2_regularizer(scale=0.001)

def conv_block(inputs,stage,nb_filter,is_training,kernel_size=3):
    x=slim.conv2d(inputs,num_outputs=nb_filter,kernel_size=kernel_size,stride=1,rate=1,activation_fn=None)
    x=slim.batch_norm(x,is_training=is_training)
    x=tf.nn.leaky_relu(x)
    x=slim.conv2d(x,num_outputs=nb_filter,kernel_size=kernel_size,stride=1,rate=1,activation_fn=None)
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
def upsample(x,nb_filter,is_training):
    is_deconv=False
    if is_deconv:
        _,_,_,C=x.get_shape().as_list()
        x=slim.conv2d_transpose(x,C,2,2)
        #x=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None)
        x=slim.batch_norm(x,is_training=is_training)
        x=tf.nn.leaky_relu(x)
    else:
        x=img_scale(x,2)
        x=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None)
        x=slim.batch_norm(x,is_training=is_training)
        x=tf.nn.leaky_relu(x)        
    return x
def Recurrent_block(input,nb_filter,is_training,t=2):
    for i in range(t):
    if i==0:
        x1=slim.conv2d(input,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None)
        x1=slim.batch_norm(x1,is_training=is_training)
        x1=tf.nn.leaky_relu(x1)
    x1=tf.add(x1,input)
    x1=slim.conv2d(x1,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None)
    x1=slim.batch_norm(x1,is_training=is_training)
    x1=tf.nn.leaky_relu(x1)
    return x1
def RRCNN_block(input,nb_filter,is_training,t=2):
    x1=slim.conv2d(input,num_outputs=nb_filter,kernel_size=1,stride=1,rate=1,activation_fn=None)
    x2=Recurrent_block(x1,nb_filter,is_training=is_training,t=t)
    x2=Recurrent_block(x2,nb_filter,is_training=is_training,t=t)
    x_add=tf.add(x1,x2)
    return x_add
def R2UNet(inputs,is_training,min_filter=16):
    nb_filter=[min_filter,min_filter*2,min_filter*4,min_filter*8,min_filter*16]
    ###encode####
    
    conv1=RRCNN_block(inputs,stage='stage_1',nb_filter=nb_filter[0],is_training=is_training)
    pool1=slim.max_pool2d(conv1,2,padding='SAME')
    
    conv2=RRCNN_block(pool1,stage='stage_1',nb_filter=nb_filter[1],is_training=is_training)
    pool2=slim.max_pool2d(conv2,2,padding='SAME')
    
    conv3=RRCNN_block(conv2,stage='stage_1',nb_filter=nb_filter[2],is_training=is_training)
    pool3=slim.max_pool2d(conv3,2,padding='SAME')
    
    conv4=RRCNN_block(pool3,stage='stage_1',nb_filter=nb_filter[3],is_training=is_training)
    pool4=slim.max_pool2d(conv4,2,padding='SAME')
    
    conv5=RRCNN_block(pool4,stage='stage_1',nb_filter=nb_filter[4],is_training=is_training)
    ###decode####
    
    up4=upsample(x,nb_filter=nb_filter[3],is_training=is_training)
    merge4=tf.concat([conv4,up4],3)
    conv4_1=conv_block(merge4,stage='stage_41',nb_filter=nb_filter[3],is_training=is_training)
    
    up3=upsample(conv4_1,nb_filter=nb_filter[3],is_training=is_training)
    merge3=tf.concat([conv3,up3],3)
    conv3_1=RRCNN_block(merge3,stage='stage_31',nb_filter=nb_filter[2],is_training=is_training)
    
    up2=upsample(conv3_1,nb_filter=nb_filter[3],is_training=is_training)
    merge2=tf.concat([conv2,up2],3)
    conv2_1=RRCNN_block(merge2,stage='stage_21',nb_filter=nb_filter[1],is_training=is_training)
    
    up1=upsample(conv2_1,nb_filter=nb_filter[3],is_training=is_training)
    merge1=tf.concat([conv1,up1],3)
    conv1_1=RRCNN_block(merge1,stage='stage_11',nb_filter=nb_filter[0],is_training=is_training)
    
    output=slim.conv2d(conv1_1,1,1,rate=1,activation_fn=tf.nn.sigmoid,scope='output')
    return output   
