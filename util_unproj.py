import tensorflow as tf
import os
import sys
import math
import numpy as np
import random
import other
import time

class Unproject_tools:
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
    
    def unproject_batch(self, invZ_batch, mask_batch, additional_batch, azimuth_batch, elevation_batch):
        depth_batch = 1.0/(invZ_batch+self.const.eps)
        depth_batch *= 2.0    
        depth_batch = depth_batch * mask_batch + self.const.DIST_TO_CAM * (1.0-mask_batch)
        inputs = tf.concat([mask_batch, depth_batch - self.const.DIST_TO_CAM, additional_batch], axis = 3)
        inputs = tf.image.resize_images(inputs, (self.const.S, self.const.S))
    
        unprojected = other.unproject.unproject(inputs)
        
        return unprojected
