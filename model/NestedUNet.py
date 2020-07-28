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
def upsample(x,nb_filter,is_training):
    _,_,_,C=x.get_shape().as_list()
    x=slim.conv2d_transpose(x,C,2,2)
    x=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None)
    x=slim.batch_norm(x,is_training=is_training)
    x=tf.nn.leaky_relu(x)
    return x
def NestedUNet(inputs,is_training,min_filter=6，deep_supervision=False):
    nb_filter=[min_filter,min_filter*2,min_filter*4,min_filter*8,min_filter*16]
   
    conv0_0=conv_block(inputs,stage='stage_00',nb_filter=nb_filter[0],is_training=is_training)
    conv1_0=conv_block(slim.max_pool2d(conv0_0,2,padding='SAME'),stage='stage_10',nb_filter=nb_filter[1],is_training=is_training)
    conv0_1=conv_block(tf.concat([conv0_0,upsample(conv1_0,nb_filter=nb_filter[0],is_training=is_training)],3),stage='stage_01',nb_filter=nb_filter[0],is_training=is_training)
    
    conv2_0=conv_block(slim.max_pool2d(conv1_0,2,padding='SAME'),stage='stage_20',nb_filter=nb_filter[2],is_training=is_training)
    conv1_1=conv_block(tf.concat([conv1_0,upsample(conv2_0,nb_filter=nb_filter[1],is_training=is_training)],3),stage='stage_11',nb_filter=nb_filter[1],is_training=is_training)
    conv0_2=conv_block(tf.concat([conv0_0,conv0_1,upsample(conv1_1,nb_filter=nb_filter[0],is_training=is_training)],3),stage='stage_02',nb_filter=nb_filter[0],is_training=is_training)
    
    conv3_0=conv_block(slim.max_pool2d(conv2_0,2,padding='SAME'),stage='stage_30',nb_filter=nb_filter[3],is_training=is_training)
    conv2_1=conv_block(tf.concat([conv2_0,upsample(conv3_0,nb_filter=nb_filter[2],is_training=is_training)],3),stage='stage_21',nb_filter=nb_filter[2],is_training=is_training)
    conv1_2=conv_block(tf.concat([conv1_0,conv1_1,upsample(conv2_1,nb_filter=nb_filter[1],is_training=is_training)],3),stage='stage_12',nb_filter=nb_filter[1],is_training=is_training)
    conv0_3=conv_block(tf.concat([conv0_0,conv0_1,conv0_2,upsample(conv1_2,nb_filter=nb_filter[0],is_training=is_training)],3),stage='stage_02',nb_filter=nb_filter[0],is_training=is_training)
    
    conv4_0=conv_block(slim.max_pool2d(conv3_0,2,padding='SAME'),stage='stage_40',nb_filter=nb_filter[4],is_training=is_training)
    conv3_1=conv_block(tf.concat([conv3_0,upsample(conv4_0,nb_filter=nb_filter[3],is_training=is_training)],3),stage='stage_31',nb_filter=nb_filter[3],is_training=is_training)
    conv2_2=conv_block(tf.concat([conv2_0,conv2_1,upsample(conv3_1,nb_filter=nb_filter[2],is_training=is_training)],3),stage='stage_22',nb_filter=nb_filter[2],is_training=is_training)
    conv1_3=conv_block(tf.concat([conv1_0,conv1_1,conv1_2,upsample(conv2_2,nb_filter=nb_filter[1],is_training=is_training)],3),stage='stage_03',nb_filter=nb_filter[1],is_training=is_training)
    conv0_4=conv_block(tf.concat([conv0_0,conv0_1,conv0_2,conv0_3,upsample(conv1_3,nb_filter=nb_filter[0],is_training=is_training)],3),stage='stage_04',nb_filter=nb_filter[0],is_training=is_training)
    
    
    output=slim.conv2d(conv0_4,1,1,rate=1,activation_fn=tf.nn.sigmoid,scope='output')
    return output   
