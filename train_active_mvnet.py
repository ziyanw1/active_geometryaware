#!/usr/bin/env python2

# scp jerrypiglet@128.237.129.33:Bitsync/3dv2017_PBA/train_ae_2_reg_lmdb.py . && scp -r jerrypiglet@128.237.129.33:Bitsync/3dv2017_PBA/models . && CUDA_VISIBLE_DEVICES=2,3 vglrun python train_ae_2_reg_lmdb.py --task_name REG_final_FASTconstantLr_bnNObn_NOtrans_car24576_bb10__bb9 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=20 --learning_rate=1e-5 --ae_file '/newfoundland/rz1/log/finalAE_1e-5_bnNObn_car24576__bb10/'
# CUDA_VISIBLE_DEVICES=0,1 vglrun python train_ae_2_reg_lmdb.py --task_name REG_final_FASTconstantLr_bnNObn_NOtrans_car24576_bb10_randLampbb8__bb9 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=20 --learning_rate=1e-5 --ae_file '/newfoundland/rz1/log/finalAE_1e-5_bnNObn_car24576__bb10/'
# scp jerrypiglet@128.237.133.169:Bitsync/3dv2017_PBA/train_ae_2_reg_lmdb.py . && scp -r jerrypiglet@128.237.133.169:Bitsync/3dv2017_PBA/models . && vglrun python train_ae_2_reg_lmdb.py --task_name REG_finalAE_FASTconstantLr_bnNObn_NOtrans_car24576_bb10__bb8_0707 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=20 --learning_rate=1e-5 --ae_file '/newfoundland/rz1/log/finalAE_FASTconstantLr_bnNObn_NOtrans_car24576__bb10'

import argparse
import time
import math
import h5py
import numpy as np
# np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
# np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc as sm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'env_data'))
import tf_util

#from visualizers import VisVox
from active_mvnet import ActiveMVnet, MVInput
from shapenet_env import ShapeNetEnv
from replay_memory import ReplayMemory, trajectData
import psutil
import gc
import resource


np.random.seed(0)
tf.set_random_seed(0)
global FLAGS
flags = tf.flags
flags.DEFINE_integer('gpu', 0, "GPU to use [default: GPU 0]")
# task and control (yellow)
flags.DEFINE_string('model_file', 'pcd_ae_1_lmdb', 'Model name')
flags.DEFINE_string('cat_name', 'airplane', 'Category name')
flags.DEFINE_string('category', '03001627', 'category Index')
flags.DEFINE_string('train_filename_prefix', 'train', '')
flags.DEFINE_string('val_filename_prefix', 'train', '')
flags.DEFINE_string('test_filename_prefix', 'train', '')
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
flags.DEFINE_integer('resolution', 128, '')
flags.DEFINE_integer('voxel_resolution', 64, '')
flags.DEFINE_string('opt_step_name', 'opt_step', '')
flags.DEFINE_string('loss_name', 'sketch_loss', '')
flags.DEFINE_integer('batch_size', 4, 'Batch Size during training [default: 32]')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate [default: 0.001]') #used to be 3e-5
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
flags.DEFINE_boolean('use_coef', False, 'if use coefficient for loss')
# log and drawing (blue)
flags.DEFINE_boolean("is_training", True, 'training flag')
flags.DEFINE_boolean("force_delete", False, "force delete old logs")
flags.DEFINE_boolean("if_summary", True, "if save summary")
flags.DEFINE_boolean("if_save", True, "if save")
flags.DEFINE_integer("save_every_step", 10, "save every ? step")
flags.DEFINE_boolean("if_test", True, "if test")
flags.DEFINE_integer("test_every_step", 2, "test every ? step")
flags.DEFINE_boolean("if_draw", True, "if draw latent")
flags.DEFINE_integer("draw_every_step", 1000, "draw every ? step")
flags.DEFINE_integer("vis_every_step", 1000, "draw every ? step")
flags.DEFINE_boolean("if_init_i", False, "if init i from 0")
flags.DEFINE_integer("init_i_to", 1, "init i to")
flags.DEFINE_integer("test_iter", 2, "init i to")
flags.DEFINE_integer("test_episode_num", 2, "init i to")
flags.DEFINE_boolean("save_test_results", True, "if init i from 0")
flags.DEFINE_boolean("if_save_eval", False, "if save evaluation results")
# reinforcement learning
flags.DEFINE_integer('mvnet_resolution', 224, 'image resolution for mvnet')
flags.DEFINE_integer('max_episode_length', 4, 'maximal episode length for each trajactory')
flags.DEFINE_integer('mem_length', 1000, 'memory length for replay memory')
flags.DEFINE_integer('action_num', 8, 'number of actions')
flags.DEFINE_integer('burn_in_length', 10, 'burn in length for replay memory')
flags.DEFINE_integer('burn_in_iter', 10, 'burn in iteration for MVnet')
flags.DEFINE_string('reward_type', 'IoU', 'reward type: [IoU, IG]')
flags.DEFINE_float('init_eps', 0.95, 'initial value for epsilon')
flags.DEFINE_float('end_eps', 0.05, 'initial value for epsilon')
flags.DEFINE_float('gamma', 0.99, 'discount factor for reward')
flags.DEFINE_string('debug_single', False, 'debug mode: using single model')
FLAGS = flags.FLAGS

#POINTCLOUDSIZE = FLAGS.num_point
#if FLAGS.if_deconv:
#    OUTPUTPOINTS = FLAGS.num_point
#else:
#    OUTPUTPOINTS = FLAGS.num_point/2
FLAGS.BN_INIT_DECAY = 0.5
FLAGS.BN_DECAY_DECAY_RATE = 0.5
FLAGS.BN_DECAY_DECAY_STEP = float(FLAGS.decay_step)
FLAGS.BN_DECAY_CLIP = 0.99

def log_string(out_str):
    FLAGS.LOG_FOUT.write(out_str+'\n')
    FLAGS.LOG_FOUT.flush()
    print(out_str)

def prepare_plot():
    plt.figure(1, figsize=(16, 32))
    plt.axis('off')
    plt.show(block=False)

    plt.figure(2, figsize=(16, 32))
    plt.axis('off')
    plt.show(block=False)

def save(ae, step, epoch, batch):
    # save_path = os.path.join(FLAGS.CHECKPOINT_DIR, FLAGS.task_name)
    log_dir = FLAGS.LOG_DIR
    ckpt_dir = os.path.join(log_dir, FLAGS.CHECKPOINT_DIR)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    saved_checkpoint = ae.saver.save(ae.sess, \
        os.path.join(ckpt_dir, 'model.ckpt'), global_step=step)
    log_string(tf_util.toBlue("-----> Model saved to file: %s; step = %d" % (saved_checkpoint, step)))

def restore(ae):
    restore_path = os.path.join(FLAGS.LOG_DIR, FLAGS.CHECKPOINT_DIR)
    latest_checkpoint = tf.train.latest_checkpoint(restore_path)
    log_string(tf_util.toYellow("----#-> Model restoring from: %s..."%restore_path))
    ae.saver.restore(ae.sess, latest_checkpoint)
    log_string(tf_util.toYellow("----- Restored from %s."%latest_checkpoint))

def restore_from_iter(ae, iter):
    restore_path = os.path.join(FLAGS.LOG_DIR, FLAGS.CHECKPOINT_DIR)
    ckpt_path = os.path.join(restore_path, 'model.ckpt-{0}'.format(iter))
    print(tf_util.toYellow("----#-> Model restoring from: {} using {} iterations...".format(restore_path, iter)))
    ae.saver.restore(ae.sess, ckpt_path)
    print(tf_util.toYellow("----- Restored from %s."%ckpt_path))

def train(active_mv):
    
    senv = ShapeNetEnv(FLAGS)
    replay_mem = ReplayMemory(FLAGS)

    #### for debug
    #a = np.array([[1,0,1],[0,0,0]])
    #b = np.array([[1,0,1],[0,1,0]])
    #print('IoU: {}'.format(replay_mem.calu_IoU(a, b)))
    #sys.exit()
    #### for debug

    log_string('====== Starting burning in memories ======')
    burn_in(senv, replay_mem)
    log_string('====== Done. {} trajectories burnt in ======'.format(FLAGS.burn_in_length))

    #epsilon = FLAGS.init_eps
    K_single = np.asarray([[420.0, 0.0, 112.0], [0.0, 420.0, 112.0], [0.0, 0.0, 1]])
    K_list = np.tile(K_single[None, None, ...], (1, FLAGS.max_episode_length, 1, 1))  

    ### burn in(pretrain) for MVnet
    if FLAGS.burn_in_iter > 0:
        for i in range(FLAGS.burn_in_iter):
            input_stuff = replay_mem.get_batch_list(FLAGS.batch_size)
            rgb_l_b = input_stuff[0]
            invz_l_b = input_stuff[1]
            mask_l_b = input_stuff[2]
            vox_b = input_stuff[3]
            azimuth_l_b = input_stuff[4]
            elevation_l_b = input_stuff[5]        
            action_l_b = input_stuff[6]

            mvnet_input = MVInput(rgb_l_b, invz_l_b, mask_l_b, azimuth_l_b, elevation_l_b, vox = vox_b, action = action_l_b)
            tic = time.time()

            out_stuff = active_mv.run_step(mvnet_input, mode = 'burnin', is_training = True)
            
            log_string('Burn in iter: {}, recon_loss: {}, unproject time: {}s'.format(i, out_stuff[2], time.time()-tic))
        #sys.exit()
        ###

    for i_idx in range(FLAGS.max_iter):
        state, model_id = senv.reset(True)
        actions = []
        RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
        invZ_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1), dtype=np.float32)
        mask_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1), dtype=np.float32)

        azimuth_temp_list = np.zeros((FLAGS.max_episode_length, 1), dtype=np.float32)
        elevation_temp_list = np.zeros((FLAGS.max_episode_length, 1), dtype=np.float32)

        
        RGB_temp_list[0, ...], mask_temp_list[0, ..., 0] = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
        invZ_temp_list[0, ..., 0] = replay_mem.read_invZ(state[0][0], state[1][0], model_id)

        azimuth_temp_list[0, 0] = state[0][0]
        elevation_temp_list[0, 0] = state[1][0]
        
        #R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
        ## TODO: 
        ## 1. forward pass for rgb and get depth/mask/sn/R use rgb2dep
        ## 2. update vox_feature_list using unproject and aggregator
        for e_idx in range(FLAGS.max_episode_length-1):
            ## processing mask
            mask_temp_list = (mask_temp_list > 0.5).astype(np.float32)
            mask_temp_list *= (invZ_temp_list >= 1e-6)
            
            tic = time.time()
            mvnet_input = MVInput(RGB_temp_list, invZ_temp_list, mask_temp_list, azimuth_temp_list, elevation_temp_list)
            agent_action = active_mv.select_action(mvnet_input, e_idx) 
            actions.append(agent_action)
            state, next_state, done, model_id = senv.step(actions[-1])
            RGB_temp_list[e_idx+1, ...], mask_temp_list[e_idx+1, ..., 0] = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)
            invZ_temp_list[e_idx+1, ..., 0] = replay_mem.read_invZ(next_state[0], next_state[1], model_id)

            azimuth_temp_list[e_idx+1, 0] = next_state[0]
            elevation_temp_list[e_idx+1, 0] = next_state[1]
            
            log_string('Iter: {}, e_idx: {}, azim: {}, elev: {}, model_id: {}, time: {}s'.format(i_idx, e_idx, next_state[0], 
                next_state[1], model_id, time.time()-tic))
            #R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
            ## TODO: update vox_temp
            ## 1. update vox_list using MVnet
            #vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
            #vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]
                ## TODO: 
                ## 1. get reward by vox_list
                #rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
                temp_traj = trajectData(traj_state, actions, model_id)
                replay_mem.append(temp_traj)
                break

        ## TODO: get batch of
        ## 1. rgb image sequence and transfer it to depth/mask/sn sequence
        ## 2. given depth/mask/sn/ sequence and transfer it to list of feature_vox using aggregator and encoder 
        ## TODO: train
        ## 1. using given sequence data and ground truth voxel, train MVnet
        ## 2. for #max_episode_length times, train aggregator and agent policy network
        input_stuff = replay_mem.get_batch_list(FLAGS.batch_size)
        rgb_l_b = input_stuff[0]
        invz_l_b = input_stuff[1]
        mask_l_b = input_stuff[2]
        vox_b = input_stuff[3]
        azimuth_l_b = input_stuff[4]
        elevation_l_b = input_stuff[5]        
        action_l_b = input_stuff[6]

        mvnet_input = MVInput(rgb_l_b, invz_l_b, mask_l_b, azimuth_l_b, elevation_l_b, vox = vox_b, action = action_l_b)
        
        tic = time.time()

        out_stuff = active_mv.run_step(mvnet_input, mode='train', is_training = True)
        
        recon_loss = out_stuff[1]
        reinforce_loss = out_stuff[2]
        summary_train = out_stuff[-1]
        mean_episode_reward = np.mean(np.sum(out_stuff[6], axis=1), axis=0)
        log_string('Iter: {}, recon_loss: {:.4f}, mean_episode_reward: {}, reinforce_loss: {}, time: {}s'.format(i_idx,
            recon_loss/(FLAGS.max_episode_length*FLAGS.batch_size), mean_episode_reward, 
            reinforce_loss, time.time()-tic))
        active_mv.train_writer.add_summary(summary_train, i_idx)
        #rgb_batch, vox_batch, reward_batch, action_batch = replay_mem.get_batch(FLAGS.batch_size)
        #print 'reward_batch: {}'.format(reward_batch)
        #print 'rewards: {}'.format(rewards)
        ## TODO train model and do logging & evaluation
        #feed_dict = {agent.is_training: True, agent.rgb_batch: rgb_batch, agent.vox_batch: vox_batch, agent.reward_batch: reward_batch,
        #    agent.action_batch: action_batch}
        #opt_train, merge_summary, loss = agent.sess.run([agent.opt, agent.merged_train, agent.loss], feed_dict=feed_dict)
        #log_string('+++++Iteration: {}, loss: {:.4f}, mean_reward: {:.4f}+++++'.format(i_idx, loss, np.mean(rewards)))
        #tf_util.save_scalar(i_idx, 'episode_total_reward', np.sum(rewards[:]), agent.train_writer) 
        #agent.train_writer.add_summary(merge_summary, i_idx)

        if i_idx % FLAGS.save_every_step == 0 and i_idx > 0:
            save(active_mv, i_idx, i_idx, i_idx) 

        if i_idx % FLAGS.test_every_step == 0 and i_idx > 0:
            eval_r_mean, eval_IoU_mean, eval_loss_mean = evaluate(active_mv, FLAGS.test_episode_num, replay_mem, i_idx)
            tf_util.save_scalar(i_idx, 'eval_mean_reward', eval_r_mean, active_mv.train_writer)
            tf_util.save_scalar(i_idx, 'eval_mean_IoU', eval_IoU_mean, active_mv.train_writer)
            tf_util.save_scalar(i_idx, 'eval_mean_loss', eval_loss_mean, active_mv.train_writer)

def evaluate(active_mv, test_episode_num, replay_mem, iter):
    senv = ShapeNetEnv(FLAGS)

    #epsilon = FLAGS.init_eps
    rewards_list = []
    IoU_list = []
    loss_list = []
    for i_idx in range(test_episode_num):
        state, model_id = senv.reset(True)
        actions = []
        RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
        invZ_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1), dtype=np.float32)
        mask_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1), dtype=np.float32)
        azimuth_temp_list = np.zeros((FLAGS.max_episode_length, 1), dtype=np.float32)
        elevation_temp_list = np.zeros((FLAGS.max_episode_length, 1), dtype=np.float32)

        #R_list = np.zeros((FLAGS.max_episode_length, 3, 4), dtype=np.float32)
        #vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
        #    dtype=np.float32)

        RGB_temp_list[0, ...], mask_temp_list[0, ..., 0] = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
        invZ_temp_list[0, ..., 0] = replay_mem.read_invZ(state[0][0], state[1][0], model_id)

        azimuth_temp_list[0, 0] = state[0][0]
        elevation_temp_list[0, 0] = state[1][0]

        ## run simulations and get memories
        for e_idx in range(FLAGS.max_episode_length-1):
            ## processing mask
            mask_temp_list = (mask_temp_list > 0.5).astype(np.float32)
            mask_temp_list *= (invZ_temp_list >= 1e-6)

            mvnet_input = MVInput(RGB_temp_list, invZ_temp_list, mask_temp_list, azimuth_temp_list, elevation_temp_list)
            active_mv_action = active_mv.select_action(mvnet_input, e_idx, is_training=False) 
            actions.append(active_mv_action)
            state, next_state, done, model_id = senv.step(actions[-1])
            
            RGB_temp_list[e_idx+1, ...], mask_temp_list[e_idx+1, ..., 0] = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)

            invZ_temp_list[e_idx+1, ..., 0] = replay_mem.read_invZ(next_state[0], next_state[1], model_id)

            azimuth_temp_list[e_idx+1, 0] = next_state[0]
            elevation_temp_list[e_idx+1, 0] = next_state[1]
            
            #R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
            ## TODO: update vox_temp
            #vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
            #vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]
                #rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
                #temp_traj = trajectData(traj_state, actions, rewards, model_id)
                break


        voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(FLAGS.category, model_id))
        vox_gt = replay_mem.read_vox(voxel_name)
        mvnet_input = MVInput(RGB_temp_list, invZ_temp_list, mask_temp_list, azimuth_temp_list, elevation_temp_list, vox = vox_gt)
        vox_final_list, recon_loss_list, rewards_test = active_mv.predict_vox_list(mvnet_input)
        
        vox_final_ = np.squeeze(vox_final_list[-1, ...])
        vox_final_[vox_final_ > 0.5] = 1
        vox_final_[vox_final_ <= 0.5] = 0
        final_IoU = replay_mem.calu_IoU(vox_final_, np.squeeze(vox_gt))
        #final_loss = replay_mem.calu_cross_entropy(vox_final_list[-1, ...], vox_gt)
        log_string('------Episode: {}, episode_reward: {:.4f}, IoU: {:.4f}, Losses: {}------'.format(
            i_idx, np.sum(rewards_test), final_IoU, recon_loss_list))
        rewards_list.append(np.sum(rewards_test))
        IoU_list.append(final_IoU)
        loss_list.append(recon_loss_list)

        if FLAGS.if_save_eval:
            save_dict = {'voxel_list': vox_final_list, 'vox_gt': vox_gt, 'model_id': model_id, 'states': traj_state,
                'RGB_list': RGB_temp_list}
            eval_dir = os.path.join(FLAGS.LOG_DIR, 'eval')
            if not os.path.exists(eval_dir):
                os.mkdir(eval_dir)
            eval_dir = os.path.join(eval_dir, '{}'.format(iter))
            if not os.path.exists(eval_dir):
                os.mkdir(eval_dir)

            mat_save_name = os.path.join(eval_dir, '{}.mat'.format(i_idx))
            sio.savemat(mat_save_name, save_dict)

    rewards_list = np.asarray(rewards_list)
    IoU_list = np.asarray(IoU_list)
    loss_list = np.asarray(loss_list)

    return np.mean(rewards_list), np.mean(IoU_list), np.mean(loss_list)

# def test(agent, test_episode_num, model_iter):
#     senv = ShapeNetEnv(FLAGS)
#     replay_mem = ReplayMemory(FLAGS)

#     K_single = np.asarray([[420.0, 0.0, 112.0], [0.0, 420.0, 112.0], [0.0, 0.0, 1]])
#     K_list = np.tile(K_single[None, None, ...], (1, FLAGS.max_episode_length, 1, 1))  
#     for i_idx in range(test_episode_num):
#         state, model_id = senv.reset(True)
#         senv.current_model = '53180e91cd6651ab76e29c9c43bc7aa'
#         senv.current_model = '41d9bd662687cf503ca22f17e86bab24'
#         model_id = senv.current_model
#         actions = []
#         RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
#         R_list = np.zeros((FLAGS.max_episode_length, 3, 4), dtype=np.float32)
#         vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
#             dtype=np.float32)

#         RGB_temp_list[0, ...], _ = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
#         R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
#         vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, 0) 
#         vox_temp = np.squeeze(vox_temp_list[0, ...])
#         ## run simulations and get memories
#         for e_idx in range(FLAGS.max_episode_length-1):
#             agent_action = agent.select_action(RGB_temp_list[e_idx], vox_temp, is_training=False)
#             actions.append(agent_action)
#             state, next_state, done, model_id = senv.step(actions[-1])
#             RGB_temp_list[e_idx+1, ...], _ = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)
#             R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
#             ## TODO: update vox_temp
#             vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
#             vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
#             if done:
#                 traj_state = state
#                 traj_state[0] += [next_state[0]]
#                 traj_state[1] += [next_state[1]]
#                 rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
#                 temp_traj = trajectData(traj_state, actions, model_id)
#                 break


#         vox_final_list = vox_temp_list

#         result_path = os.path.join(FLAGS.LOG_DIR, 'results')
#         if not os.path.exists(result_path):
#             os.mkdir(result_path)

#         if FLAGS.save_test_results:
#             result_path_iter = os.path.join(result_path, '{}'.format(model_iter))
#             if not os.path.exists(result_path_iter):
#                 os.mkdir(result_path_iter)
        
#         voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(FLAGS.category, model_id))
#         vox_gt = replay_mem.read_vox(voxel_name)

#         mat_path = os.path.join(result_path_iter, '{}.mat'.format(i_idx))
#         sio.savemat(mat_path, {'vox_list':vox_final_list, 'vox_gt': vox_gt, 'RGB': RGB_temp_list, 'model_id': model_id, 'states': traj_state})


#         #log_string('+++++Iteration: {}, loss: {}, mean_reward: {}+++++'.format(i_idx, loss, np.mean(rewards)))

def burn_in(senv, replay_mem):     
    K_single = np.asarray([[420.0, 0.0, 112.0], [0.0, 420.0, 112.0], [0.0, 0.0, 1]])
    K_list = np.tile(K_single[None, None, ...], (1, FLAGS.max_episode_length, 1, 1))  
    tic = time.time()
    for i_idx in range(FLAGS.burn_in_length):
        if i_idx % 10 == 0 and i_idx != 0:
            toc = time.time()
            log_string('Burning in {}/{} sequences, time taken: {}s'.format(i_idx, FLAGS.burn_in_length, toc-tic))
            tic = time.time()
        state, model_id = senv.reset(True)
        actions = []
        RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
        R_list = np.zeros((FLAGS.max_episode_length, 3, 4), dtype=np.float32)
        #vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
        #    dtype=np.float32)

        RGB_temp_list[0, ...], _ = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
        R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
        #vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, 0) 
        #vox_temp = np.squeeze(vox_temp_list[0, ...])
        ## run simulations and get memories
        for e_idx in range(FLAGS.max_episode_length-1):
            actions.append(np.random.randint(FLAGS.action_num))
            state, next_state, done, model_id = senv.step(actions[-1])
            RGB_temp_list[e_idx+1, ...], _ = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)
            R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
            ## TODO: update vox_temp
            #vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
            #vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]
                #rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
                #print 'rewards: {}'.format(rewards)
                temp_traj = trajectData(traj_state, actions, model_id)
                replay_mem.append(temp_traj)
                break

if __name__ == "__main__":
    #MODEL = importlib.import_module(FLAGS.model_file) # import network module
    #MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model_file+'.py')
   
    ####### log writing
    FLAGS.LOG_DIR = FLAGS.LOG_DIR + '/' + FLAGS.task_name
    #FLAGS.CHECKPOINT_DIR = os.path.join(FLAGS.CHECKPOINT_DIR, FLAGS.task_name)
    #tf_util.mkdir(FLAGS.CHECKPOINT_DIR)

    if not FLAGS.is_training:
        agent = ActiveAgent(FLAGS)
        restore_from_iter(agent, FLAGS.test_iter) 
        test(agent, FLAGS.test_episode_num, FLAGS.test_iter)

        sys.exit()

    if not os.path.exists(FLAGS.LOG_DIR):
        os.mkdir(FLAGS.LOG_DIR)
        print tf_util.toYellow('===== Created %s.'%FLAGS.LOG_DIR)
    else:
        # os.system('rm -rf %s/*'%FLAGS.LOG_DIR)
        if not(FLAGS.restore):
            
            def check_delete():
                if FLAGS.force_delete:
                    return True
                delete_key = raw_input(tf_util.toRed('===== %s exists. Delete? [y (or enter)/N] '%FLAGS.LOG_DIR))
                return delete_key == 'y' or delete_key == ''
            
            if check_delete():
                os.system('rm -rf %s/*'%FLAGS.LOG_DIR)
                #os.system('rm -rf %s/*'%FLAGS.CHECKPOINT_DIR)
                print tf_util.toRed('Deleted.'+FLAGS.LOG_DIR)
            else:
                print tf_util.toRed('Overwrite.')
        else:
            print tf_util.toRed('To Be Restored...')

    #tf_util.mkdir(os.path.join(FLAGS.LOG_DIR, 'saved_images'))
    ##os.system('cp %s %s' % (MODEL_FILE, FLAGS.LOG_DIR)) # bkp of model def
    ##os.system('cp train.py %s' % (FLAGS.LOG_DIR)) # bkp of train procedure


    FLAGS.LOG_FOUT = open(os.path.join(FLAGS.LOG_DIR, 'log_train.txt'), 'w')
    FLAGS.LOG_FOUT.write(str(FLAGS)+'\n')

    ##prepare_plot()
    #log_string(tf_util.toYellow('<<<<'+FLAGS.task_name+'>>>> '+str(tf.flags.FLAGS.__flags)))
    ##### log writing
    active_mv = ActiveMVnet(FLAGS)
    #if FLAGS.restore:
    #    restore(ae)
    train(active_mv)

    # z_list = []
    # test_demo_render_z(ae, z_list)

    FLAGS.LOG_FOUT.close()
