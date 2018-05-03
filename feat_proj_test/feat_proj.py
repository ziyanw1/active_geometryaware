#!/usr/bin/env python2

import sys
sys.path.append('..')

from env_data.shapenet_env import ShapeNetEnv, trajectData
from env_data.replay_memory import ReplayMemory
from models.active_agent import ActiveAgent
import tensorflow as tf
import os
import math
import numpy as np
import random
import other
import time

unp = other.unproject

t0 = time.time()

const = other.constants

execfile('flag_stuff.py')
FLAGS.batch_size = 1

#########

const.S = FLAGS.voxel_resolution
const.RESOLUTION = FLAGS.resolution

const.DIST_TO_CAM = 4.0
const.NEAR_PLANE = 3.0
const.FAR_PLANE = 5.0

fov = 30.0
focal_length = 1.0/math.tan(fov*math.pi/180/2)
const.focal_length = focal_length
const.BS = None

const.DEBUG_UNPROJECT = False
const.USE_LOCAL_BIAS = False
const.USE_OUTLINE = True
        
const.mode = 'train'
const.rpvx_unsup = False
const.force_batchnorm_trainmode = False
const.force_batchnorm_testmode = False
const.NET3DARCH = 'marr_64'
const.eps = 1e-6

#########

model_id = 'fed8ee6ce00ab015d8f27b2e727c3511'

env = ShapeNetEnv(FLAGS)
mem = ReplayMemory(FLAGS)
#agent = ActiveAgent(FLAGS)

state0, _ = env.reset(True)

ACTION = 5
for i in range(5):
    _, state1, _, _ = env.step(ACTION)

voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(FLAGS.category, model_id))

az0 = state0[0][0]
el0 = state0[1][0]
az1 = state1[0]
el1 = state1[1]

rgb0, mask0 = mem.read_png_to_uint8(az0, el0, model_id)
invz0 = mem.read_invZ(az0, el0, model_id)
invz0_ = mem.read_invZ(az0, el0, model_id, resize = False)

rgb1, mask1 = mem.read_png_to_uint8(az1, el1, model_id)
invz1 = mem.read_invZ(az1, el1, model_id)

vox = mem.read_vox(voxel_name, transpose = False)
#be sure not to transpose while reading in voxels

vox = np.expand_dims(vox, axis = 0)
vox = np.expand_dims(vox, axis = 4)

rotation_obj = unp.make_rotation_object(az0, el0, az1, el1)

rotation_obj = unp.stack_rotation_objects([rotation_obj])

gt_rotation = unp.make_rotation_object(az0, el0, 0.0, 0.0)

#stuff everything into tensors and do all required data preprocessing

def make_tensors_for_raw_inputs(rgb, invz, mask):
    mask = (mask > 0.5).astype(np.float32)
    mask *= (invz >= const.eps)
    
    invz = np.expand_dims(invz, axis = 2)
    mask = np.expand_dims(mask, axis = 2)

    rgb = np.expand_dims(rgb, axis = 0)
    mask = np.expand_dims(mask, axis = 0)
    invz = np.expand_dims(invz, axis = 0)

    #getshape = lambda x: (None,) + x.shape[1:]
    getshape = lambda x: (1,) + x.shape[1:]
    make_ph = lambda x: tf.placeholder(shape = getshape(x), dtype = tf.float32)

    rgb_ = make_ph(rgb)
    invz_ = make_ph(invz)
    mask_ = make_ph(mask)

    depth = 1.0/(invz_+const.eps)
    depth *= 2.0    
    depth = depth * mask_ + const.DIST_TO_CAM * (1.0-mask_)

    feed_dict = {rgb_: rgb.copy(), invz_: invz.copy(), mask_: mask.copy()}

    return rgb_, depth, mask_, feed_dict

rgb0, depth0, mask0, fd = make_tensors_for_raw_inputs(rgb0, invz0, mask0)
rgb1, depth1, mask1, fd1 = make_tensors_for_raw_inputs(rgb1, invz1, mask1)

gt_vox = tf.placeholder(shape = vox.shape, dtype = tf.float32)

fd.update({gt_vox : vox})
fd.update(fd1)

vox = tf.constant(vox, dtype = tf.float32)

###########

out0 = unp.unproject_and_rotate(depth0, mask0, rgb0, None)
out1 = unp.unproject_and_rotate(depth1, mask1, rgb1, None)
out2 = unp.unproject_and_rotate(depth1, mask1, rgb1, rotation_obj)

#these can be fed into the voxel net as follows:
prediction = other.nets.voxel_net_3d(out0)

#we can visualize the outlines pretty well
outline_idx = int(out0.get_shape()[-1])-1
outline0 = out0[:,:,:,:,outline_idx:outline_idx+1]
outline1 = out1[:,:,:,:,outline_idx:outline_idx+1]
outline2 = out2[:,:,:,:,outline_idx:outline_idx+1]

####
#also rotate ground truth voxels to the pose of state 0
gt_vox = other.voxel.transformer_preprocess(gt_vox)
gt_vox = other.voxel.rotate_voxel(gt_vox, gt_rotation[0])
gt_vox = other.voxel.rotate_voxel(gt_vox, gt_rotation[1])

#### some postprocessing, so that we can view the unprojected voxels and check for reasonableness

out_depth0, out_mask0 = unp.flatten(unp.project_and_postprocess(outline0))
out_depth1, out_mask1 = unp.flatten(unp.project_and_postprocess(outline1))
out_depth2, out_mask2 = unp.flatten(unp.project_and_postprocess(outline2))
out_depth3, out_mask3 = unp.flatten(unp.project_and_postprocess(gt_vox))

ops_to_run = {
    'depth0': depth0,
    'mask0': mask0,
    'rgb0': rgb0,
    'out_depth0': out_depth0,
    'out_mask0': out_mask0,
    'depth1': depth1,
    'mask1': mask1,
    'rgb1': rgb1,
    'out_depth1': out_depth1,
    'out_mask1': out_mask1,
    'out_depth2': out_depth2,
    'out_mask2': out_mask2,
    'out_depth3': out_depth3,
    'out_mask3': out_mask3,
}

#other.img.imsave('debug/invz0.png', invz0)
#other.img.imsave('debug/invz1.png', invz1)

print 'built graph in %f seconds' % (time.time()-t0)
t0 = time.time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for i in range(10):
    start = time.time()
    outputs = sess.run(ops_to_run, feed_dict = fd)
    end = time.time()
    print 'iteration %d took %f s' % (i, end-start)

print 'finished in %f seconds' % (time.time()-t0)
t0 = time.time()

print '\ndepth0'
print np.min(outputs['depth0'])
print np.max(outputs['depth0'])

print '\nout_depth0'
print np.min(outputs['out_depth0'])
print np.max(outputs['out_depth0'])

print '\nout_depth2'
print np.min(outputs['out_depth2'])
print np.max(outputs['out_depth2'])

print '\nout_depth3'
print np.min(outputs['out_depth3'])
print np.max(outputs['out_depth3'])

print '\ndepth1'
print np.min(outputs['depth1'])
print np.max(outputs['depth1'])

print '\nout_depth1'
print np.min(outputs['out_depth1'])
print np.max(outputs['out_depth1'])


other.img.imsave01('debug/depth0.png', outputs['depth0'][0,:,:,0], const.NEAR_PLANE, const.FAR_PLANE)
other.img.imsave01('debug/mask0.png', outputs['mask0'][0,:,:,0])
other.img.imsave01('debug/rgb0.png', outputs['rgb0'][0])
other.img.imsave01('debug/out_depth0.png', outputs['out_depth0'][0,:,:,0], const.NEAR_PLANE, const.FAR_PLANE)
other.img.imsave01('debug/out_mask0.png', outputs['out_mask0'][0,:,:,0])

other.img.imsave01('debug/depth1.png', outputs['depth1'][0,:,:,0], const.NEAR_PLANE, const.FAR_PLANE)
other.img.imsave01('debug/mask1.png', outputs['mask1'][0,:,:,0])
other.img.imsave01('debug/rgb1.png', outputs['rgb1'][0])
other.img.imsave01('debug/out_depth1.png', outputs['out_depth1'][0,:,:,0], const.NEAR_PLANE, const.FAR_PLANE)
other.img.imsave01('debug/out_mask1.png', outputs['out_mask1'][0,:,:,0])

other.img.imsave01('debug/out_depth2.png', outputs['out_depth2'][0,:,:,0], const.NEAR_PLANE, const.FAR_PLANE)
other.img.imsave01('debug/out_mask2.png', outputs['out_mask2'][0,:,:,0])

other.img.imsave01('debug/out_depth3.png', outputs['out_depth3'][0,:,:,0], const.NEAR_PLANE, const.FAR_PLANE)
other.img.imsave01('debug/out_mask3.png', outputs['out_mask3'][0,:,:,0])

print 'dumped outputs in %f seconds' % (time.time()-t0)
