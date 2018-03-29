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
        other.const = other.constants
        other.const.S = FLAGS.voxel_resolution
        
        other.const.DIST_TO_CAM = 4.0
        other.const.NEAR_PLANE = 3.0
        other.const.FAR_PLANE = 5.0
        
        fov = 30.0
        focal_length = 1.0/math.tan(fov*math.pi/180/2)
        other.const.focal_length = focal_length
        other.const.BS = FLAGS.batch_size*FLAGS.max_episode_length
        
        other.const.DEBUG_UNPROJECT = False
        other.const.USE_LOCAL_BIAS = False
        other.const.USE_OUTLINE = True
                
        other.const.mode = 'train'
        other.const.rpvx_unsup = False
        other.const.force_batchnorm_trainmode = False
        other.const.force_batchnorm_testmode = False
        other.const.NET3DARCH = 'marr'
        other.const.eps = 1e-6
    
        #########

    def unproject(self, invZ, mask, additional, azimuth, elevation):
        depth = 1.0/(invZ+other.const.eps)
        depth *= 2.0
        depth = depth * mask + other.const.DIST_TO_CAM * (1.0-mask)

        rots = other.unproject.rotate_to_first(azimuth[:,:,0], elevation[:,:,0])
        depths = tf.unstack(depth, axis = 1)
        masks = tf.unstack(mask, axis = 1)
        additionals = tf.unstack(additional, axis = 1)
        
        unprojects = [
            other.unproject.unproject_and_rotate(depth_, mask_, additional_, rot_)
            for (depth_, mask_, additional_, rot_) in zip(depths, masks, additionals, rots)
        ]

        return tf.stack(unprojects, axis = 1) #4 x 4 x V x V x V x 7
