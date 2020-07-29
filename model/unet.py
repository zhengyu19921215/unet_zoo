import tensorflow as tf 


def conv_block(inputs,stage,nb_filter,is_training,kernel_size=3):
    x=tf.layers.conv2d(inputs,filters=nb_filter,kernel_size=kernel_size,stride=1,padding='same')
    x=tf.layers.batch_normalization(x,is_training=is_training)
    x=tf.nn.leaky_relu(x)
    x=tf.layers.conv2d(x,filters=nb_filter,kernel_size=kernel_size,stride=1,padding='same')
    x=tf.layers.batch_normalization(x,is_training=is_training)
    x=tf.nn.leaky_relu(x)
    return x
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
        #_,_,_,C=x.get_shape().as_list()
        x=tf.layers.conv2d_transpose(x,nb_filter,2,2,padding='same')       
        x=tf.layers.batch_normalization(x,is_training=is_training)
        x=tf.nn.leaky_relu(x)
    else:
        x=img_scale(x,2)
        x=tf.layers.conv2d(x,filters=nb_filter,kernel_size=3,stride=1,padding='same')
        x=tf.layers.batch_normalization(x,is_training=is_training)
        x=tf.nn.leaky_relu(x)      
    return x
def UNet(inputs,is_training,min_filter=6):
    nb_filter=[min_filter,min_filter*2,min_filter*4,min_filter*8,min_filter*16]
    ###encode####
    
    conv1=conv_block(inputs,stage='stage_1',nb_filter=nb_filter[0],is_training=is_training)
    pool1=tf.layers.max_pooling2d(conv1,2,2,padding='same')
    
    conv2=conv_block(pool1,stage='stage_1',nb_filter=nb_filter[1],is_training=is_training)
    pool2=tf.layers.max_pooling2d(conv2,2,2,padding='same')
    
    conv3=conv_block(conv2,stage='stage_1',nb_filter=nb_filter[2],is_training=is_training)
    pool3=tf.layers.max_pooling2d(conv3,2,2,padding='same')
    
    conv4=conv_block(pool3,stage='stage_1',nb_filter=nb_filter[3],is_training=is_training)
    pool4=tf.layers.max_pooling2d(conv4,2,2,padding='same')
    
    conv5=conv_block(pool4,stage='stage_1',nb_filter=nb_filter[4],is_training=is_training)
    ###decode####
    
    up4=upsample(x,nb_filter=nb_filter[3],is_training=is_training)
    merge4=tf.concat([conv4,up4],3)
    conv4_1=conv_block(merge4,stage='stage_41',nb_filter=nb_filter[3],is_training=is_training)
    
    up3=upsample(conv4_1,nb_filter=nb_filter[3],is_training=is_training)
    merge3=tf.concat([conv3,up3],3)
    conv3_1=conv_block(merge3,stage='stage_31',nb_filter=nb_filter[2],is_training=is_training)
    
    up2=upsample(conv3_1,nb_filter=nb_filter[3],is_training=is_training)
    merge2=tf.concat([conv2,up2],3)
    conv2_1=conv_block(merge2,stage='stage_21',nb_filter=nb_filter[1],is_training=is_training)
    
    up1=upsample(conv2_1,nb_filter=nb_filter[3],is_training=is_training)
    merge1=tf.concat([conv1,up1],3)
    conv1_1=conv_block(merge1,stage='stage_11',nb_filter=nb_filter[0],is_training=is_training)
    
    output=tf.layers.conv2d(conv1_1,1,1,rate=1,activation=tf.nn.sigmoid,scope='output')
    return output   
