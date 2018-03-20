#!/usr/bin/env python2

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

t0 = time.time()

const = other.constants

global FLAGS
flags = tf.flags

# task and control (yellow)
flags.DEFINE_string('model_file', 'pcd_ae_1_lmdb', 'Model name')
flags.DEFINE_string('cat_name', 'airplane', 'Category name')
flags.DEFINE_string('category', '03001627', 'category Index')
#flags.DEFINE_string('LOG_DIR', '/newfoundland/rz1/log/summary', 'Log dir [default: log]')
flags.DEFINE_string('LOG_DIR', './log_agent', 'Log dir [default: log]')
flags.DEFINE_string('data_path', './data/lmdb', 'data directory')
flags.DEFINE_string('data_file', 'rgb2depth_single_0212', 'data file')
#flags.DEFINE_string('CHECKPOINT_DIR', '/newfoundland/rz1/log', 'Log dir [default: log]')
flags.DEFINE_string('CHECKPOINT_DIR', './log_agent', 'Log dir [default: log]')
flags.DEFINE_integer('max_ckpt_keeps', 10, 'maximal keeps for ckpt file [default: 10]')
flags.DEFINE_string('task_name', 'tmp', 'task name to create under /LOG_DIR/ [default: tmp]')
flags.DEFINE_boolean('restore', False, 'If resume from checkpoint')
flags.DEFINE_string('ae_file', '', '')
# train (green)
flags.DEFINE_integer('num_point', 2048, 'Point Number [256/512/1024/2048] [default: 1024]')
flags.DEFINE_integer('resolution', 224, '')
flags.DEFINE_integer('voxel_resolution', 128, '')
flags.DEFINE_string('opt_step_name', 'opt_step', '')
flags.DEFINE_string('loss_name', 'sketch_loss', '')
flags.DEFINE_integer('batch_size', 16, 'Batch Size during training [default: 32]')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate [default: 0.001]') #used to be 3e-5
flags.DEFINE_float('momentum', 0.95, 'Initial learning rate [default: 0.9]')
flags.DEFINE_string('optimizer', 'adam', 'adam or momentum [default: adam]')
flags.DEFINE_integer('decay_step', 5000000, 'Decay step for lr decay [default: 200000]')
flags.DEFINE_float('decay_rate', 0.7, 'Decay rate for lr decay [default: 0.8]')
flags.DEFINE_integer('max_iter', 1000000, 'Decay step for lr decay [default: 200000]')
# arch (magenta)
flags.DEFINE_string('network_name', 'ae', 'Name for network architecture used for rgb to depth')
flags.DEFINE_boolean('if_deconv', True, 'If add deconv output to generator aside from fc output')
flags.DEFINE_boolean('if_constantLr', True, 'If use constant lr instead of decaying one')
flags.DEFINE_boolean('if_en_bn', True, 'If use batch normalization for the mesh decoder')
flags.DEFINE_boolean('if_gen_bn', False, 'If use batch normalization for the mesh generator')
flags.DEFINE_boolean('if_bn', True, 'If use batch normalization for the mesh decoder')
flags.DEFINE_float('bn_decay', 0.95, 'Decay rate for batch normalization [default: 0.9]')
flags.DEFINE_boolean("if_transform", False, "if use two transform layers")
flags.DEFINE_float('reg_weight', 0.1, 'Reweight for mat loss [default: 0.1]')
flags.DEFINE_boolean("if_vae", False, "if use VAE instead of vanilla AE")
flags.DEFINE_boolean("if_l2Reg", False, "if use l2 regularizor for the generator")
flags.DEFINE_float('vae_weight', 0.1, 'Reweight for mat loss [default: 0.1]')
flags.DEFINE_boolean('use_gan', False, 'if using GAN [default: False]')
# log and drawing (blue)
flags.DEFINE_boolean("is_training", True, 'training flag')
flags.DEFINE_boolean("force_delete", False, "force delete old logs")
flags.DEFINE_boolean("if_summary", True, "if save summary")
flags.DEFINE_boolean("if_save", True, "if save")
flags.DEFINE_integer("save_every_step", 2000, "save every ? step")
flags.DEFINE_boolean("if_test", True, "if test")
flags.DEFINE_integer("test_every_step", 2000, "test every ? step")
flags.DEFINE_boolean("if_draw", True, "if draw latent")
flags.DEFINE_integer("draw_every_step", 1000, "draw every ? step")
flags.DEFINE_integer("vis_every_step", 1000, "draw every ? step")
flags.DEFINE_boolean("if_init_i", False, "if init i from 0")
flags.DEFINE_integer("init_i_to", 1, "init i to")
flags.DEFINE_integer("test_iter", 2, "init i to")
flags.DEFINE_integer("test_episode_num", 20, "init i to")
flags.DEFINE_boolean("save_test_results", True, "if init i from 0")
# reinforcement learning
flags.DEFINE_integer('mvnet_resolution', 224, 'image resolution for mvnet')
flags.DEFINE_integer('max_episode_length', 5, 'maximal episode length for each trajactory')
flags.DEFINE_integer('mem_length', 10000000, 'memory length for replay memory')
flags.DEFINE_integer('action_num', 8, 'number of actions')
flags.DEFINE_integer('burn_in_length', 10000, 'burn in length for replay memory')
flags.DEFINE_string('reward_type', 'IoU', 'reward type: [IoU, IG]')
flags.DEFINE_float('init_eps', 0.95, 'initial value for epsilon')
flags.DEFINE_float('end_eps', 0.05, 'initial value for epsilon')
flags.DEFINE_float('gamma', 0.99, 'discount factor for reward')
flags.DEFINE_string('debug_single', False, 'debug mode: using single model')
FLAGS = flags.FLAGS

FLAGS.batch_size = 1

#########

const.S = FLAGS.voxel_resolution

const.DIST_TO_CAM = 4.0
const.NEAR_PLANE = 3.0
const.FAR_PLANE = 5.0

fov = 30.0
focal_length = 1.0/math.tan(fov*math.pi/180/2)
const.focal_length = focal_length
const.BS = FLAGS.batch_size

const.DEBUG_UNPROJECT = False
const.USE_LOCAL_BIAS = False
const.USE_OUTLINE = True
        
const.mode = 'train'
const.rpvx_unsup = False
const.force_batchnorm_trainmode = False
const.force_batchnorm_testmode = False
const.NET3DARCH = 'marr'
const.eps = 1e-6

#########

model_id = 'fed8ee6ce00ab015d8f27b2e727c3511'

env = ShapeNetEnv(FLAGS)
mem = ReplayMemory(FLAGS)
#agent = ActiveAgent(FLAGS)

state0, _ = env.reset(True)

ACTION = 5
for i in range(3):
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

vox = mem.read_vox(voxel_name)
vox = np.expand_dims(vox, axis = 0)

def make_rotation_object(az0, el0, az1, el1):
    #returns object which can be used to rotate view 1 -> 0
    #see vs/nets.translate_views for reference implementation

    dtheta = az1 - az0

    r1 = other.voxel.get_transform_matrix(theta = 0.0, phi = -el1)
    r2 = other.voxel.get_transform_matrix(theta = dtheta, phi = el0)
    
    return (r1, r2)

rotation_obj = make_rotation_object(az0, el0, az1, el1)

#stuff everything into tensors and do all required data preprocessing

def make_tensors_for_raw_inputs(rgb, invz, mask):
    mask = (mask > 128.0).astype(np.float32)
    mask *= (invz >= const.eps)
    
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

    depth = 1.0/(invz_+const.eps)
    depth *= 2.0    
    depth = depth * mask_ + const.DIST_TO_CAM * (1.0-mask_)

    feed_dict = {rgb_: rgb.copy(), invz_: invz.copy(), mask_: mask.copy()}

    return rgb_, depth, mask_, feed_dict

rgb0, depth0, mask0, fd = make_tensors_for_raw_inputs(rgb0, invz0, mask0)
rgb1, depth1, mask1, fd1 = make_tensors_for_raw_inputs(rgb1, invz1, mask1)

fd.update(fd1)

vox = tf.constant(vox, dtype = tf.float32)

###########

#shapes:

#depth: BS x H x W x 1
#mask: BS x H x W x 1
#additional: BS x H x W x K

def unproject_and_rotate(depth, mask, additional, rotation = None):

    #order of concate is important
    inputs = tf.concat([mask, depth - const.DIST_TO_CAM, additional], axis = 3)
    inputs = tf.image.resize_images(inputs, (const.S, const.S))

    unprojected = other.nets.unproject(inputs)

    if rotation is not None:
        rotated = other.voxel.rotate_voxel(unprojected, rotation[0])
        rotated = other.voxel.rotate_voxel(rotated, rotation[1])
    else:
        rotated = unprojected

    return rotated

out0 = unproject_and_rotate(depth0, mask0, rgb0, None)
out1 = unproject_and_rotate(depth1, mask1, rgb1, None)
out2 = unproject_and_rotate(depth1, mask1, rgb1, rotation_obj)

#we can visualize the outlines pretty well
outline_idx = int(out0.get_shape()[-1])-1
outline0 = out0[:,:,:,:,outline_idx:outline_idx+1]
outline1 = out1[:,:,:,:,outline_idx:outline_idx+1]
outline2 = out2[:,:,:,:,outline_idx:outline_idx+1]

#### some postprocessing, so that we can view the unprojected voxels and check for reasonableness

proj_and_post = lambda x: other.voxel.transformer_postprocess(other.voxel.project_voxel(x))

def flatten(voxels):
    H = FLAGS.resolution
    W = FLAGS.resolution            
            
    pred_depth = other.voxel.voxel2depth_aligned(voxels)
    pred_mask = other.voxel.voxel2mask_aligned(voxels)
        
    #replace bg with grey
    hard_mask = tf.cast(pred_mask > 0.5, tf.float32)
    pred_depth *= hard_mask

    pred_depth += const.DIST_TO_CAM * (1.0 - hard_mask)

    pred_depth = tf.image.resize_images(pred_depth, (H, W))
    pred_mask = tf.image.resize_images(pred_mask, (H, W))
    return pred_depth, pred_mask

out_depth0, out_mask0 = flatten(proj_and_post(outline0))
out_depth1, out_mask1 = flatten(proj_and_post(outline1))
out_depth2, out_mask2 = flatten(proj_and_post(outline2))

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

print 'dumped outputs in %f seconds' % (time.time()-t0)
