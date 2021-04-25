"""
# Author :zhengyu
# Date: 25/04/2021
Purpose of file:
               1):Read tfrecords and generate dataset ,interation and batch tensor
"""
import tensorflow as tf
from utils.augmentation import random_augmenattion

def parse_function(filenames,img_shape,mask_shape,is_training):
    features={
              "image":tf.io.FixedLenFeature([img_shape[0]*img_shape[1]*img_shape[2]],tf.float32),
              "mask":tf.io.FixedLenFeature([mask_shape[0]*mask_shape[1]*mask_shape[2]],tf.float32)
              }
    ## image
    image=parsed_example["image"]
    image=tf.reshape(image,image_shape)
    
    ## mask
    
    mask=parsed_example["mask"]
    mask=tf.reshape(mask,mask_shape)
    
    if is_training:
      img,mask=random_augmenattion(image,mask)
    image=tf.ccast(image,tf.float32)
    mask=tf.ccast(mask,tf.float32) 
    return image,mask
def make_batch_iterator(tfrecord_path,img_shape,mask_shape,nums,is_training=True,batch_size=16):
    def _parse_fn(tfrecord):
        return parse_function(tfrecord,img_shape,mask_shape,is_training)
    if is_training:
        dataset=dataset.shuffle(nums)
    dataset=dataset.map(_parse_fn,num_parallel_calls=4)
    dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(buffer_size=3*batch_size)
    iterator=dataset.make_initializable_iterator()
    next_batch=iterator.get_next()
    return dataset,iterator,next_batch
    
