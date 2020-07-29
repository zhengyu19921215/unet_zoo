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
def Attention_block(g,x,nb_filter,is_training):
    conv_g=slim.conv2d(g,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None,weigt_regularizer=reg)
    conv_g=slim.batch_norm(conv_g,is_training=is_training)
    
    conv_x=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None,weigt_regularizer=reg)
    conv_x=slim.batch_norm(conv_x,is_training=is_training)
    
    conv_add=tf.add(conv_g,conv_x)
    conv_add=tf.nn.leaky_relu(x)
    
    conv_add=slim.conv2d(conv_add,1,1,1,rate=1,,activation_fn=None,weigt_regularizer=reg)
    conv_add=slim.batch_norm(conv_add,is_training=is_training)
    conv_add=tf.nn.sigmoid(conv_add)
    final=tf.multiply(x,conv_add)
    return final
    
    
def Attention_UNet(inputs,is_training,min_filter=6):
    nb_filter=[min_filter,min_filter*2,min_filter*4,min_filter*8,min_filter*16]
    ###encode####
    
    conv1=conv_block(inputs,stage='stage_1',nb_filter=nb_filter[0],is_training=is_training)
    pool1=slim.max_pool2d(conv1,2,padding='SAME')
    
    conv2=conv_block(pool1,stage='stage_1',nb_filter=nb_filter[1],is_training=is_training)
    pool2=slim.max_pool2d(conv2,2,padding='SAME')
    
    conv3=conv_block(conv2,stage='stage_1',nb_filter=nb_filter[2],is_training=is_training)
    pool3=slim.max_pool2d(conv3,2,padding='SAME')
    
    conv4=conv_block(pool3,stage='stage_1',nb_filter=nb_filter[3],is_training=is_training)
    pool4=slim.max_pool2d(conv4,2,padding='SAME')
    
    conv5=conv_block(pool4,stage='stage_1',nb_filter=nb_filter[4],is_training=is_training)
    ###decode####
    
    up4=upsample(conv5,nb_filter=nb_filter[3],is_training=is_training)
    Att4=Attention_block(up4,conv4,nb_filter[3],is_training=is_training)
    merge4=tf.concat([Att4,up4],3)
    conv4_1=conv_block(merge4,stage='stage_41',nb_filter=nb_filter[3],is_training=is_training)
    
    up3=upsample(conv4_1,nb_filter=nb_filter[2],is_training=is_training)
    Att3=Attention_block(up3,conv3,nb_filter[2],is_training=is_training)
    merge3=tf.concat([Att3,up3],3)
    conv3_1=conv_block(merge3,stage='stage_31',nb_filter=nb_filter[3],is_training=is_training)
    
    up2=upsample(conv3_1,nb_filter=nb_filter[1],is_training=is_training)
    Att2=Attention_block(up2,conv2,nb_filter[1],is_training=is_training)
    merge2=tf.concat([Att2,up2],3)
    conv2_1=conv_block(merge2,stage='stage_21',nb_filter=nb_filter[1],is_training=is_training)
    
    up1=upsample(conv2_1,nb_filter=nb_filter[0],is_training=is_training)
    Att1=Attention_block(up1,conv1,nb_filter[0],is_training=is_training)
    merge1=tf.concat([Att1,up1],3)
    conv1_1=conv_block(merge1,stage='stage_11',nb_filter=nb_filter[0],is_training=is_training)
    
    output=slim.conv2d(conv1_1,1,1,rate=1,activation_fn=tf.nn.sigmoid,scope='output')
    return output   
