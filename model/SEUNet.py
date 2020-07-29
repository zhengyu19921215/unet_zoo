import tensorflow as tf 
import numpy as np
import tensorflow.contrib.slim as slim
import os
reg=slim.l2_regularizer(scale=0.001)

def SElayer(x,reduction):
    _,H,W,C=x.get_shape().as_list()
    y=slim.avg_pool2d(x,[H,W],stride=1,scope='global_avg_pool')
    y=tf.layers.dense(y,units=C//reduction)
    y=tf.nn.leaky_relu(y)
    y=tf.layers.dense(y,units=C,activation=tf.nn.sigmoid)
    return x**y
def SEBlock(x,nb_filter,stage,reduction,is_training):
    out=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None,weight_regularizer=reg)
    out=slim.batch_norm(out,is_training=is_training)
    out=tf.nn.leaky_relu(out)
    out=slim.conv2d(out,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None,weight_regularizer=reg)
    out=slim.batch_norm(out,is_training=is_training)
    out=SElayer(x,reduction)
    out=tf.nn.leaky_relu(out)
    return out
def SEResidualBlock(x,nb_filter,stage,reduction,is_training):
    residual=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None,weight_regularizer=reg)
    residual=slim.batch_norm(residual,is_training=is_training)
    
    out=slim.conv2d(x,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None,weight_regularizer=reg)
    out=slim.batch_norm(out,is_training=is_training)
    out=tf.nn.leaky_relu(out)
    out=slim.conv2d(out,num_outputs=nb_filter,kernel_size=3,stride=1,rate=1,activation_fn=None,weight_regularizer=reg)
    out=slim.batch_norm(out,is_training=is_training)
    out=SElayer(x,reduction)
    out+=residual
    out=tf.nn.leaky_relu(out)
    return out


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
def SEUNet(inputs,is_training,min_filter=16,min_reduction=4):
   
    nb_filter=[min_filter,min_filter*2,min_filter*4,min_filter*8,min_filter*16]
    nb_reduction=[min_reduction,min_reduction*2,min_reduction*4]
    ###encode####
    
    conv1=SEBlock(inputs,nb_filter[0],stage='stage_1',reduction=nb_reduction[0],is_training=is_training)
    pool1=slim.max_pool2d(conv1,2,padding='SAME')
    
    conv2=SEBlock(pool1,nb_filter[1],stage='stage_1',reduction=nb_reduction[1],is_training=is_training)
    pool2=slim.max_pool2d(conv2,2,padding='SAME')
    
    conv3=SEBlock(pool2,nb_filter[2],stage='stage_1',reduction=nb_reduction[1],is_training=is_training)
    pool3=slim.max_pool2d(conv3,2,padding='SAME')
    
    conv4=SEBlock(pool3,nb_filter[3],stage='stage_1',reduction=nb_reduction[2],is_training=is_training)
    pool4=slim.max_pool2d(conv4,2,padding='SAME')
     
    conv5=SEBlock(pool4,nb_filter[4],stage='stage_1',reduction=nb_reduction[2],is_training=is_training)
    ####decode
    
    up4=upsample(x,nb_filter=nb_filter[3],is_training=is_training)
    merge4=tf.concat([conv4,up4],3)
    conv4_1=SEBlock(merge4,stage='stage_41',reduction=nb_reduction[2],nb_filter=nb_filter[3],is_training=is_training)
    
    up3=upsample(conv4_1,nb_filter=nb_filter[2],is_training=is_training)
    merge3=tf.concat([conv3,up3],3)
    conv3_1=SEBlock(merge3,stage='stage_31',reduction=nb_reduction[1],nb_filter=nb_filter[2],is_training=is_training)
    
    up2=upsample(conv3_1,nb_filter=nb_filter[1],is_training=is_training)
    merge2=tf.concat([conv2,up2],3)
    conv2_1=SEBlock(merge2,stage='stage_21',reduction=nb_reduction[1],nb_filter=nb_filter[1],is_training=is_training)
    
    up1=upsample(conv2_1,nb_filter=nb_filter[0],is_training=is_training)
    merge1=tf.concat([conv1,up1],3)
    conv1_1=SEBlock(merge1,stage='stage_11',reduction=nb_reduction[0],nb_filter=nb_filter[0],is_training=is_training)
    
    output=slim.conv2d(conv1_1,1,1,rate=1,activation_fn=tf.nn.sigmoid,scope='output')
    return output   
def SEResidualUNet(inputs,is_training,min_filter=16,min_reduction=4):
   
    nb_filter=[min_filter,min_filter*2,min_filter*4,min_filter*8,min_filter*16]
    nb_reduction=[min_reduction,min_reduction*2,min_reduction*4]
    ###encode####
    
    conv1=SEResidualBlock(inputs,nb_filter[0],stage='stage_1',reduction=nb_reduction[0],is_training=is_training)
    pool1=slim.max_pool2d(conv1,2,padding='SAME')
    
    conv2=SEResidualBlock(pool1,nb_filter[1],stage='stage_1',reduction=nb_reduction[1],is_training=is_training)
    pool2=slim.max_pool2d(conv2,2,padding='SAME')
    
    conv3=SEResidualBlock(pool2,nb_filter[2],stage='stage_1',reduction=nb_reduction[1],is_training=is_training)
    pool3=slim.max_pool2d(conv3,2,padding='SAME')
    
    conv4=SEResidualBlock(pool3,nb_filter[3],stage='stage_1',reduction=nb_reduction[2],is_training=is_training)
    pool4=slim.max_pool2d(conv4,2,padding='SAME')
     
    conv5=SEBlock(pool4,nb_filter[4],stage='stage_1',reduction=nb_reduction[2],is_training=is_training)
    ####decode
    
    up4=upsample(x,nb_filter=nb_filter[3],is_training=is_training)
    merge4=tf.concat([conv4,up4],3)
    conv4_1=SEResidualBlock(merge4,stage='stage_41',reduction=nb_reduction[2],nb_filter=nb_filter[3],is_training=is_training)
    
    up3=upsample(conv4_1,nb_filter=nb_filter[2],is_training=is_training)
    merge3=tf.concat([conv3,up3],3)
    conv3_1=SEResidualBlock(merge3,stage='stage_31',reduction=nb_reduction[1],nb_filter=nb_filter[2],is_training=is_training)
    
    up2=upsample(conv3_1,nb_filter=nb_filter[1],is_training=is_training)
    merge2=tf.concat([conv2,up2],3)
    conv2_1=SEResidualBlock(merge2,stage='stage_21',reduction=nb_reduction[1],nb_filter=nb_filter[1],is_training=is_training)
    
    up1=upsample(conv2_1,nb_filter=nb_filter[0],is_training=is_training)
    merge1=tf.concat([conv1,up1],3)
    conv1_1=SEResidualBlock(merge1,stage='stage_11',reduction=nb_reduction[0],nb_filter=nb_filter[0],is_training=is_training)
    
    output=slim.conv2d(conv1_1,1,1,rate=1,activation_fn=tf.nn.sigmoid,scope='output')
    return output   
