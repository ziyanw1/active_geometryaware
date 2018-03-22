import tensorflow as tf
import os
import sys
import math
import numpy as np
import random
import other
import time

class unproject_tools:
    def __init__(self, FLAGS):
        #########
        self.const = other.constants
        self.const.S = FLAGS.voxel_resolution
        
        self.const.DIST_TO_CAM = 4.0
        self.const.NEAR_PLANE = 3.0
        self.const.FAR_PLANE = 5.0
        
        fov = 30.0
        focal_length = 1.0/math.tan(fov*math.pi/180/2)
        self.const.focal_length = focal_length
        self.const.BS = FLAGS.batch_size*FLAGS.max_episode_length
        
        self.const.DEBUG_UNPROJECT = False
        self.const.USE_LOCAL_BIAS = False
        self.const.USE_OUTLINE = True
                
        self.const.mode = 'train'
        self.const.rpvx_unsup = False
        self.const.force_batchnorm_trainmode = False
        self.const.force_batchnorm_testmode = False
        self.const.NET3DARCH = 'marr'
        self.const.eps = 1e-6
    
        #########
    
    def make_rotation_object(self, az0, el0, az1, el1):
        #returns object which can be used to rotate view 1 -> 0
        #see vs/nets.translate_views for reference implementation
    
        dtheta = az1 - az0
    
        r1 = other.voxel.get_transform_matrix(theta = 0.0, phi = -el1)
        r2 = other.voxel.get_transform_matrix(theta = dtheta, phi = el0)
        
        return (r1, r2)
    
    def unproject_and_rotate(self, depth, mask, additional, rotation = None):
    
        #order of concate is important
        inputs = tf.concat([mask, depth - self.const.DIST_TO_CAM, additional], axis = 3)
        inputs = tf.image.resize_images(inputs, (self.const.S, self.const.S))
    
        unprojected = other.unproject.unproject(inputs)
    
        if rotation is not None:
            rotated = other.voxel.rotate_voxel(unprojected, rotation[0])
            rotated = other.voxel.rotate_voxel(rotated, rotation[1])
        else:
            rotated = unprojected
    
        return rotated
    
    def make_tensors_for_raw_inputs(self, rgb, invz, mask):
        mask = (mask > 0.5).astype(np.float32)
        mask *= (invz >= self.const.eps)
        
        invz = np.expand_dims(invz, axis = 2)
        mask = np.expand_dims(mask, axis = 2)
    
        rgb = np.expand_dims(rgb, axis = 0)
        mask = np.expand_dims(mask, axis = 0)
        invz = np.expand_dims(invz, axis = 0)
    
        rgb_ = tf.placeholder(shape = rgb.shape, dtype = tf.float32)
        invz_ = tf.placeholder(shape = invz.shape, dtype = tf.float32)
        mask_ = tf.placeholder(shape = mask.shape, dtype = tf.float32)
        #rgb = tf.constant(rgb, dtype = tf.float32)
        #invz = tf.constant(invz, dtype = tf.float32)
        #mask = tf.constant(mask, dtype = tf.float32)
    
        depth = 1.0/(invz_+self.const.eps)
        depth *= 2.0    
        depth = depth * mask_ + self.const.DIST_TO_CAM * (1.0-mask_)
    
        feed_dict = {rgb_: rgb.copy(), invz_: invz.copy(), mask_: mask.copy()}
    
        return rgb_, depth, mask_, feed_dict
    
    def unproject_batch(self, invZ_batch, mask_batch, additional_batch, azimuth_batch, elevation_batch):
        depth_batch = 1.0/(invZ_batch+self.const.eps)
        depth_batch *= 2.0    
        depth_batch = depth_batch * mask_batch + self.const.DIST_TO_CAM * (1.0-mask_batch)
        inputs = tf.concat([mask_batch, depth_batch - self.const.DIST_TO_CAM, additional_batch], axis = 3)
        inputs = tf.image.resize_images(inputs, (self.const.S, self.const.S))
    
        unprojected = other.unproject.unproject(inputs)
        
        return unprojected
