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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'env_data'))
import tf_util

#from visualizers import VisVox
from active_agent import ActiveAgent
from shapenet_env import ShapeNetEnv, trajectData
from replay_memory import ReplayMemory
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
flags.DEFINE_integer('voxel_resolution', 32, '')
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

def select_action(agent, rgb, vox, is_training=True):
    feed_dict = {agent.is_training:False, agent.rgb_batch: rgb[None, ...], agent.vox_batch: vox[None, ...]}
    
    #if np.random.uniform(low=0.0, high=1.0) > epsilon:
    #    action_prob = agent.sess.run([agent.action_prob], feed_dict=feed_dict)
    #else:
    #    return np.random.randint(low=0, high=FLAGS.action_num)
    stuff = agent.sess.run([agent.action_prob], feed_dict=feed_dict)
    action_prob = stuff[0][0]
    if is_training:
        a_response = np.random.choice(action_prob, p=action_prob)

        a_idx = np.argmax(action_prob == a_response)
    else:
        a_idx = np.argmax(action_prob)
    return a_idx

def train(agent):
    
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
    for i_idx in range(FLAGS.max_iter):
        state, model_id = senv.reset(True)
        actions = []
        RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
        R_list = np.zeros((FLAGS.max_episode_length, 3, 4), dtype=np.float32)
        vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
            dtype=np.float32)

        RGB_temp_list[0, ...], _ = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
        R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
        vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, 0) 
        vox_temp = np.squeeze(vox_temp_list[0, ...])
        ## run simulations and get memories
        for e_idx in range(FLAGS.max_episode_length-1):
            agent_action = select_action(agent, RGB_temp_list[e_idx], vox_temp) 
            actions.append(agent_action)
            state, next_state, done, model_id = senv.step(actions[-1])
            RGB_temp_list[e_idx+1, ...], _ = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)
            R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
            ## TODO: update vox_temp
            vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
            vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]
                rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
                temp_traj = trajectData(traj_state, actions, rewards, model_id)
                replay_mem.append(temp_traj)
                break


        rgb_batch, vox_batch, reward_batch, action_batch = replay_mem.get_batch(FLAGS.batch_size)
        #print 'reward_batch: {}'.format(reward_batch)
        #print 'rewards: {}'.format(rewards)
        feed_dict = {agent.is_training: True, agent.rgb_batch: rgb_batch, agent.vox_batch: vox_batch, agent.reward_batch: reward_batch,
            agent.action_batch: action_batch}
        opt_train, merge_summary, loss = agent.sess.run([agent.opt, agent.merged_train, agent.loss], feed_dict=feed_dict)
        log_string('+++++Iteration: {}, loss: {:.4f}, mean_reward: {:.4f}+++++'.format(i_idx, loss, np.mean(rewards)))
        tf_util.save_scalar(i_idx, 'episode_total_reward', np.sum(rewards[:]), agent.train_writer) 
        agent.train_writer.add_summary(merge_summary, i_idx)

        if i_idx % FLAGS.save_every_step == 0 and i_idx > 0:
            save(agent, i_idx, i_idx, i_idx) 

        if i_idx % FLAGS.test_every_step == 0 and i_idx > 0:
            eval_r_mean, eval_IoU_mean, eval_loss_mean = evaluate(agent, FLAGS.test_episode_num, replay_mem)
            tf_util.save_scalar(i_idx, 'eval_mean_reward', eval_r_mean, agent.train_writer)
            tf_util.save_scalar(i_idx, 'eval_mean_IoU', eval_IoU_mean, agent.train_writer)
            tf_util.save_scalar(i_idx, 'eval_mean_loss', eval_loss_mean, agent.train_writer)

def evaluate(agent, test_episode_num, replay_mem):
    senv = ShapeNetEnv(FLAGS)

    #epsilon = FLAGS.init_eps
    K_single = np.asarray([[420.0, 0.0, 112.0], [0.0, 420.0, 112.0], [0.0, 0.0, 1]])
    K_list = np.tile(K_single[None, None, ...], (1, FLAGS.max_episode_length, 1, 1))  
    rewards_list = []
    IoU_list = []
    loss_list = []
    for i_idx in range(test_episode_num):
        state, model_id = senv.reset(True)
        actions = []
        RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
        R_list = np.zeros((FLAGS.max_episode_length, 3, 4), dtype=np.float32)
        vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
            dtype=np.float32)

        RGB_temp_list[0, ...], _ = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
        R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
        vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, 0) 
        vox_temp = np.squeeze(vox_temp_list[0, ...])
        ## run simulations and get memories
        for e_idx in range(FLAGS.max_episode_length-1):
            agent_action = select_action(agent, RGB_temp_list[e_idx], vox_temp, is_training=False) 
            actions.append(agent_action)
            state, next_state, done, model_id = senv.step(actions[-1])
            RGB_temp_list[e_idx+1, ...], _ = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)
            R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
            ## TODO: update vox_temp
            vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
            vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]
                rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
                #temp_traj = trajectData(traj_state, actions, rewards, model_id)
                break


        vox_final_list = np.squeeze(vox_temp_list)
        voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(FLAGS.category, model_id))
        vox_gt = replay_mem.read_vox(voxel_name)
        vox_final_ = vox_final_list[-1, ...]
        vox_final_[vox_final_ > 0.5] = 1
        vox_final_[vox_final_ <= 0.5] = 0
        final_IoU = replay_mem.calu_IoU(vox_final_, vox_gt)
        final_loss = replay_mem.calu_cross_entropy(vox_final_list[-1, ...], vox_gt)
        log_string('------Episode: {}, episode_reward: {:.4f}, IoU: {:.4f}, Loss: {:.4f}------'.format(
            i_idx, np.sum(rewards), final_IoU, final_loss))
        rewards_list.append(np.sum(rewards))
        IoU_list.append(final_IoU)
        loss_list.append(final_loss)

    rewards_list = np.asarray(rewards_list)
    IoU_list = np.asarray(IoU_list)
    loss_list = np.asarray(loss_list)

    return np.mean(rewards_list), np.mean(IoU_list), np.mean(loss_list)

def test(agent, test_episode_num, model_iter):
    senv = ShapeNetEnv(FLAGS)
    replay_mem = ReplayMemory(FLAGS)

    K_single = np.asarray([[420.0, 0.0, 112.0], [0.0, 420.0, 112.0], [0.0, 0.0, 1]])
    K_list = np.tile(K_single[None, None, ...], (1, FLAGS.max_episode_length, 1, 1))  
    for i_idx in range(test_episode_num):
        state, model_id = senv.reset(True)
        senv.current_model = '53180e91cd6651ab76e29c9c43bc7aa'
        senv.current_model = '41d9bd662687cf503ca22f17e86bab24'
        model_id = senv.current_model
        actions = []
        RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
        R_list = np.zeros((FLAGS.max_episode_length, 3, 4), dtype=np.float32)
        vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
            dtype=np.float32)

        RGB_temp_list[0, ...], _ = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
        R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
        vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, 0) 
        vox_temp = np.squeeze(vox_temp_list[0, ...])
        ## run simulations and get memories
        for e_idx in range(FLAGS.max_episode_length-1):
            agent_action = select_action(agent, RGB_temp_list[e_idx], vox_temp, is_training=False) 
            actions.append(agent_action)
            state, next_state, done, model_id = senv.step(actions[-1])
            RGB_temp_list[e_idx+1, ...], _ = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)
            R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
            ## TODO: update vox_temp
            vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
            vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]
                rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
                temp_traj = trajectData(traj_state, actions, rewards, model_id)
                break


        vox_final_list = vox_temp_list

        result_path = os.path.join(FLAGS.LOG_DIR, 'results')
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        if FLAGS.save_test_results:
            result_path_iter = os.path.join(result_path, '{}'.format(model_iter))
            if not os.path.exists(result_path_iter):
                os.mkdir(result_path_iter)
        
        voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(FLAGS.category, model_id))
        vox_gt = replay_mem.read_vox(voxel_name)

        mat_path = os.path.join(result_path_iter, '{}.mat'.format(i_idx))
        sio.savemat(mat_path, {'vox_list':vox_final_list, 'vox_gt': vox_gt, 'RGB': RGB_temp_list, 'model_id': model_id, 'states': traj_state})


        #log_string('+++++Iteration: {}, loss: {}, mean_reward: {}+++++'.format(i_idx, loss, np.mean(rewards)))

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
        vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
            dtype=np.float32)

        RGB_temp_list[0, ...], _ = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
        R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
        vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, 0) 
        vox_temp = np.squeeze(vox_temp_list[0, ...])
        ## run simulations and get memories
        for e_idx in range(FLAGS.max_episode_length-1):
            actions.append(np.random.randint(FLAGS.action_num))
            state, next_state, done, model_id = senv.step(actions[-1])
            RGB_temp_list[e_idx+1, ...], _ = replay_mem.read_png_to_uint8(next_state[0], next_state[1], model_id)
            R_list[e_idx+1, ...] = replay_mem.get_R(next_state[0], next_state[1])
            ## TODO: update vox_temp
            vox_temp_list = replay_mem.get_vox_pred(RGB_temp_list, R_list, K_list, e_idx+1) 
            vox_temp = np.squeeze(vox_temp_list[e_idx+1, ...])
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]
                rewards = replay_mem.get_seq_rewards(RGB_temp_list, R_list, K_list, model_id)
                #print 'rewards: {}'.format(rewards)
                temp_traj = trajectData(traj_state, actions, rewards, model_id)
                replay_mem.append(temp_traj)
                break

    #for i in range(FLAGS.max_episode_length*FLAGS.burn_in_length):
    #    actions = []
    #    RGB_temp_list = np.zeros((FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3), dtype=np.float32)
    #    R_list = np.zeros((FLAGS.max_episode_length, 3, 4), dtype=np.float32)
    #    vox_temp = np.zeros((FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution),
    #        dtype=np.float32)

    #    epsilon = FLAGS.end_eps + (FLAGS.init_eps-FLAGS.end_eps)*i_idx / FLAGS.max_iter

    #    RGB_temp_list[0, ...], _ = replay_mem.read_png_to_uint8(state[0][0], state[1][0], model_id)
    #    R_list[0, ...] = replay_mem.get_R(state[0][0], state[1][0])
    #    actions.append(np.random.randint(0,8))
    #    state, next_state, done, model = senv.step(actions[-1])
    #    if done:
    #        traj_state = state
    #        traj_state[0] += [next_state[0]]
    #        traj_state[1] += [next_state[1]]
    #        temp_traj = trajectData(traj_state, actions, model)
    #        #print temp_traj.states, temp_traj.actions, temp_traj.model_id
    #        replay_mem.append(temp_traj)
    #        senv.reset(True)
    #        actions = []

         
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
    agent = ActiveAgent(FLAGS)
    #if FLAGS.restore:
    #    restore(ae)
    train(agent)

    # z_list = []
    # test_demo_render_z(ae, z_list)

    FLAGS.LOG_FOUT.close()
