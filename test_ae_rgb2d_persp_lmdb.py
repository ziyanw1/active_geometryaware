#!/usr/bin/env python2

# scp jerrypiglet@128.237.129.33:Bitsync/3dv2017_PBA/train_ae_2_reg_lmdb.py . && scp -r jerrypiglet@128.237.129.33:Bitsync/3dv2017_PBA/models . && CUDA_VISIBLE_DEVICES=2,3 vglrun python train_ae_2_reg_lmdb.py --task_name REG_final_FASTconstantLr_bnNObn_NOtrans_car24576_bb10__bb9 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=20 --learning_rate=1e-5 --ae_file '/newfoundland/rz1/log/finalAE_1e-5_bnNObn_car24576__bb10/'
# CUDA_VISIBLE_DEVICES=0,1 vglrun python train_ae_2_reg_lmdb.py --task_name REG_final_FASTconstantLr_bnNObn_NOtrans_car24576_bb10_randLampbb8__bb9 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=20 --learning_rate=1e-5 --ae_file '/newfoundland/rz1/log/finalAE_1e-5_bnNObn_car24576__bb10/'
# scp jerrypiglet@128.237.133.169:Bitsync/3dv2017_PBA/train_ae_2_reg_lmdb.py . && scp -r jerrypiglet@128.237.133.169:Bitsync/3dv2017_PBA/models . && vglrun python train_ae_2_reg_lmdb.py --task_name REG_finalAE_FASTconstantLr_bnNObn_NOtrans_car24576_bb10__bb8_0707 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=20 --learning_rate=1e-5 --ae_file '/newfoundland/rz1/log/finalAE_FASTconstantLr_bnNObn_NOtrans_car24576__bb10'

import argparse
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
import tf_util
import scipy.misc as sm

#from visualizers import VisVox
from ae_rgb2depth_test import AE_rgb2d
import psutil
import gc
import resource

sys.path.append(os.path.join(BASE_DIR, 'data'))
from write_lmdb_rgbAndDepth import read_png_to_uint8, get_surface_from_depth

np.random.seed(0)
tf.set_random_seed(0)
global FLAGS
flags = tf.flags
flags.DEFINE_integer('gpu', 0, "GPU to use [default: GPU 0]")
# task and control (yellow)
flags.DEFINE_string('model_file', 'pcd_ae_1_lmdb', 'Model name')
flags.DEFINE_string('cat_name', 'airplane', 'Category name')
#flags.DEFINE_string('LOG_DIR', '/newfoundland/rz1/log/summary', 'Log dir [default: log]')
flags.DEFINE_string('LOG_DIR', './log/', 'Log dir [default: log]')
flags.DEFINE_string('data_path', './data/lmdb', 'data directory')
flags.DEFINE_string('data_file', 'rgb2depth_single_0212', 'data file')
flags.DEFINE_string('test_list_path', 'data/render_scripts/lists/03001627_debug.txt', 'test file path')
flags.DEFINE_string('test_data_dir', 'data/data_cache/blender_renderings/03001627/res128_chair_all', 'test data directory')
#flags.DEFINE_string('CHECKPOINT_DIR', '/newfoundland/rz1/log', 'Log dir [default: log]')
flags.DEFINE_string('CHECKPOINT_DIR', 'ckpt', 'Log dir [default: log]')
flags.DEFINE_integer('max_ckpt_keeps', 10, 'maximal keeps for ckpt file [default: 10]')
flags.DEFINE_string('task_name', 'rgb2depth_0218_unet', 'task name to create under /LOG_DIR/ [default: tmp]')
flags.DEFINE_boolean('restore', True, 'If resume from checkpoint')
flags.DEFINE_string('ae_file', '', '')
# train (green)
flags.DEFINE_integer('num_point', 2048, 'Point Number [256/512/1024/2048] [default: 1024]')
flags.DEFINE_integer('resolution', 128, '')
flags.DEFINE_integer('voxel_resolution', 32, '')
flags.DEFINE_string('opt_step_name', 'opt_step', '')
flags.DEFINE_string('loss_name', 'sketch_loss', '')
flags.DEFINE_integer('batch_size', 16, 'Batch Size during training [default: 32]')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate [default: 0.001]') #used to be 3e-5
flags.DEFINE_float('momentum', 0.95, 'Initial learning rate [default: 0.9]')
flags.DEFINE_string('optimizer', 'adam', 'adam or momentum [default: adam]')
flags.DEFINE_integer('decay_step', 5000000, 'Decay step for lr decay [default: 200000]')
flags.DEFINE_float('decay_rate', 0.7, 'Decay rate for lr decay [default: 0.8]')
flags.DEFINE_integer('max_iter', 1000000, 'Decay step for lr decay [default: 200000]')
# arch (magenta)
flags.DEFINE_string('network_name', 'unet', 'Name for network architecture used for rgb to depth')
flags.DEFINE_boolean('if_deconv', True, 'If add deconv output to generator aside from fc output')
flags.DEFINE_boolean('if_constantLr', True, 'If use constant lr instead of decaying one')
flags.DEFINE_boolean('if_en_bn', True, 'If use batch normalization for the mesh decoder')
flags.DEFINE_boolean('if_gen_bn', False, 'If use batch normalization for the mesh generator')
flags.DEFINE_float('bn_decay', 0.95, 'Decay rate for batch normalization [default: 0.9]')
flags.DEFINE_boolean("if_transform", False, "if use two transform layers")
flags.DEFINE_float('reg_weight', 0.1, 'Reweight for mat loss [default: 0.1]')
flags.DEFINE_boolean("if_vae", False, "if use VAE instead of vanilla AE")
flags.DEFINE_boolean("if_l2Reg", False, "if use l2 regularizor for the generator")
flags.DEFINE_float('vae_weight', 0.1, 'Reweight for mat loss [default: 0.1]')
# log and drawing (blue)
flags.DEFINE_boolean("force_delete", False, "force delete old logs")
flags.DEFINE_boolean("if_summary", True, "if save summary")
flags.DEFINE_boolean("if_save", True, "if save")
flags.DEFINE_integer("save_every_step", 1000, "save every ? step")
flags.DEFINE_boolean("if_test", True, "if test")
flags.DEFINE_integer("test_every_step", 5000, "test every ? step")
flags.DEFINE_boolean("if_draw", True, "if draw latent")
flags.DEFINE_integer("draw_every_step", 1000, "draw every ? step")
flags.DEFINE_integer("vis_every_step", 1000, "draw every ? step")
flags.DEFINE_boolean("if_init_i", False, "if init i from 0")
flags.DEFINE_integer("init_i_to", 1, "init i to")
flags.DEFINE_string('save_result_path', 'results/tmp', 'path to save results')
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

azim_all = np.linspace(0, 360, 9)
azim_all = azim_all[0:-1]
elev_all = np.linspace(-30, 30, 5)
VIEWS = len(azim_all)*len(elev_all)

def log_string(out_str):
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
        os.path.join(ckpt_dir, 'step%d-epoch%d-batch%d.ckpt' % (step, epoch, batch)), \
        global_step=step)
    log_string(tf_util.toBlue("-----> Model saved to file: %s; step = %d" % (saved_checkpoint, step)))

def restore(ae):
    restore_path = os.path.join(FLAGS.LOG_DIR, FLAGS.CHECKPOINT_DIR)
    print restore_path
    latest_checkpoint = tf.train.latest_checkpoint(restore_path)
    log_string(tf_util.toYellow("----#-> Model restoring from: %s..."%restore_path))
    ae.restorer.restore(ae.sess, latest_checkpoint)
    log_string(tf_util.toYellow("----- Restored from %s."%latest_checkpoint))

def train(ae):

    #v = VisVox()

    ae.opt_step = getattr(ae, FLAGS.opt_step_name)
    ae.loss_tensor = getattr(ae, FLAGS.loss_name)
    
    i = 0 
    try:
        while not ae.coord.should_stop():
            ae.sess.run(ae.assign_i_op, feed_dict={ae.set_i_to_pl: i})

            tic = time.time()
            feed_dict = {ae.is_training: True, ae.data_loader.is_training: True}

            ops_to_run = [
                ae.opt_step, ae.merge_train, ae.counter, ae.loss_tensor,
                ae.depth_recon_loss, ae.sn_recon_loss, ae.mask_cls_loss]

            stuff = ae.sess.run(ops_to_run, feed_dict = feed_dict)
            opt, summary, step, loss, depth_recon_loss, sn_recon_loss, mask_cls_loss = stuff
            toc = time.time()

            log_string('Iteration: {} time {}, loss: {}, depth_recon_loss: {}, sn_recon_loss {}, mask_cls_loss {}'.format(i, \
                toc-tic, loss, depth_recon_loss, sn_recon_loss, mask_cls_loss))
            log_string(' maxrss: {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            #gc.collect()

            print 'cpu: {}, vmem: {}, avai: {}'.format(psutil.cpu_percent(), psutil.virtual_memory().used >> 30,
                psutil.virtual_memory().available >> 30)

            i += 1
            ae.train_writer.add_summary(summary, i)
            ae.train_writer.flush()

            if i%FLAGS.save_every_step == 0:
                save(ae, i, i, i)

            if i%FLAGS.test_every_step == 0:
                test_losses = test(ae)  
                for key, value in test_losses.iteritems():
                    tf_util.save_scalar(i, 'test/'+key, value, ae.train_writer)

            gc.collect()

            #if i%FLAGS.vis_every_step == 0:
            #    v.process(vis, 'train', i)
            
            if i > FLAGS.max_iter:
                print('Done training')
                break
    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        ae.coord.request_stop()

    ae.coord.join(ae.threads)
    ae.sess.close()

def process_invZ(invZ):
    depth = np.reciprocal(invZ)
    depth[np.where(np.abs(depth)>5)] = 2.0
    depth = depth - 1.5
    #depth[np.where(depth<=0)] = 0
    #depth[np.where(depth>=1.0)] = 1.0

    return depth

def compare_a_b(a, b, dir_path, azim, elev, mode):
    
    color_map = 'gray'
    path = os.path.join(dir_path, '{}_{}_{}.png'.format(mode, int(azim), int(elev)))
    if mode == 'sn':
        color_map=None

    plt.subplot(121)
    plt.imshow(a, cmap=color_map, vmin=0, vmax=1)
    plt.subplot(122)
    plt.imshow(b, cmap=color_map, vmin=0, vmax=1)
    plt.savefig(path)



def test(ae, test_list_path):
    
    test_idx = 0
    log_string(tf_util.toGreen('=============Testing============='))
    loss = []
    depth_losses = []
    sn_losses = []
    mask_losses = []
    ae.loss_tensor = getattr(ae, FLAGS.loss_name)
    results_dir = FLAGS.save_result_path
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    with open(test_list_path, 'r') as f:
        model_names = f.readlines()

    toc = time.time()
    plt.figure()

    for model_name in model_names:
        model_directory = os.path.join(FLAGS.test_data_dir, model_name[:-1])
        model_save_dir = os.path.join(results_dir, model_name[:-1])
        depth_save_dir = os.path.join(model_save_dir, 'depth')
        mask_save_dir = os.path.join(model_save_dir, 'mask')
        sn_save_dir = os.path.join(model_save_dir, 'sn')
        compare_save_dir = os.path.join(model_save_dir, 'compare')

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        if not os.path.exists(depth_save_dir):
            os.mkdir(depth_save_dir)
        if not os.path.exists(mask_save_dir):
            os.mkdir(mask_save_dir)
        if not os.path.exists(sn_save_dir):
            os.mkdir(sn_save_dir)
        if not os.path.exists(compare_save_dir):
            os.mkdir(compare_save_dir)
        
        for a in azim_all:
            for e in elev_all:
                image_name = os.path.join(model_directory, 'RGB_{}_{}.png'.format(int(a), int(e)))
                invZ_name = os.path.join(model_directory, 'invZ_{}_{}.npy'.format(int(a), int(e)))
                invZ_save_npy = os.path.join(depth_save_dir, 'invZ_{}_{}.npy'.format(int(a), int(e)))
                invZ_save_png = os.path.join(depth_save_dir, 'invZ_{}_{}.png'.format(int(a), int(e)))
                mask_save_png = os.path.join(mask_save_dir, 'mask_{}_{}.png'.format(int(a), int(e)))
                mask_save_npy = os.path.join(mask_save_dir, 'mask_{}_{}.npy'.format(int(a), int(e)))
                sn_save_npy = os.path.join(sn_save_dir, 'sn_{}_{}.npy'.format(int(a), int(e)))
                sn_save_png = os.path.join(sn_save_dir, 'sn_{}_{}.png'.format(int(a), int(e)))
                rgb_single, mask_single_gt = read_png_to_uint8(image_name)
                invZ_single_gt = np.load(invZ_name)
                sn_single_gt = get_surface_from_depth(invZ_single_gt)
                #print mask_single.shape
                #print mask_single.dtype
                if invZ_single_gt.ndim == 2:
                    invZ_single_gt = invZ_single_gt[None, :, :, None]
                if mask_single_gt.ndim == 2:
                    mask_single_gt = mask_single_gt[None, :, :, None]
                if mask_single_gt.shape[2] == 3:
                    mask_single_gt = mask_single_gt[:, :, 0]
                    mask_single_gt = mask_single_gt[None, :, :, None]
                if rgb_single.ndim == 3:
                    rgb_single = rgb_single[None, ...]
                if sn_single_gt.ndim == 3:
                    sn_single_gt = sn_single_gt[None, ...]

                feed_dict = {ae.is_training:False, ae.rgb_batch: rgb_single, ae.invZ_batch: invZ_single_gt,
                    ae.mask_batch: mask_single_gt, ae.sn_batch: sn_single_gt}
                
                ops_to_run = [
                    ae.invZ_pred, ae.mask_pred, ae.sn_pred, ae.loss_tensor,
                    ae.depth_recon_loss, ae.sn_recon_loss, ae.mask_cls_loss, ae.global_i]
                
                stuff = ae.sess.run(ops_to_run, feed_dict = feed_dict)
                invZ_pred, mask_pred, sn_pred, loss, depth_recon_loss, sn_recon_loss, mask_cls_loss, global_step = stuff
                depth_losses.append(depth_recon_loss)
                sn_losses.append(sn_recon_loss)
                mask_losses.append(mask_cls_loss)
                print global_step

                invZ_pred = np.squeeze(np.asarray(invZ_pred, dtype=np.float32), axis=[0,-1])
                mask_pred = np.squeeze(np.asarray(mask_pred, dtype=np.float32), axis=[0,-1])
                sn_pred = np.squeeze(np.asarray(sn_pred, dtype=np.float32), axis=0)

                np.save(invZ_save_npy, invZ_pred)
                sm.imsave(invZ_save_png, invZ_pred) 
                np.save(mask_save_npy, mask_pred)
                sm.imsave(mask_save_png, mask_pred)
                np.save(sn_save_npy, sn_pred)
                sm.imsave(sn_save_png, sn_pred)

                invZ_single = np.squeeze(invZ_single_gt)
                Z_single = process_invZ(invZ_single)
                Z_pred = process_invZ(invZ_pred)
                plt.subplot(121)
                plt.imshow(Z_single, cmap='gray', vmin=0, vmax=1)
                plt.subplot(122)
                plt.imshow(Z_pred, cmap='gray', vmin=0, vmax=1)
                #plt.show()
                compare_a_b(np.squeeze(invZ_single_gt), invZ_pred, compare_save_dir, a, e, mode='invZ')
                compare_a_b(np.squeeze(mask_single_gt), mask_pred, compare_save_dir, a, e, mode='mask')
                compare_a_b(np.squeeze(sn_single_gt), sn_pred, compare_save_dir, a, e, mode='sn')


    
    tic = time.time()
    mean_depth_loss = np.mean(np.asarray(depth_losses))
    mean_sn_loss = np.mean(np.asarray(sn_losses))
    mean_mask_loss = np.mean(np.asarray(mask_losses))
    log_string(tf_util.toRed('Test time {}s, depth recon loss: {}, sn recon loss: {}, mask cls loss:{}.'.format(\
        toc-tic, mean_depth_loss, mean_sn_loss, mean_mask_loss)))

    #try:
    #    while not ae.coord.should_stop():
            #ae.sess.run(ae.assign_i_op, feed_dict={ae.set_i_to_pl: i})
    #for test_idx in range(1500):
    #    tic = time.time()
    #    feed_dict = {ae.is_training: False, ae.data_loader.is_training: False}

    #    ops_to_run = [
    #        ae.opt_step, ae.merge_train, ae.counter, ae.loss_tensor,
    #        ae.depth_recon_loss, ae.sn_recon_loss, ae.mask_cls_loss]

    #    stuff = ae.sess.run(ops_to_run, feed_dict = feed_dict)
    #    invZ_pred, mask_pred, sn_pred, loss, depth_recon_loss, sn_recon_loss, mask_cls_loss = stuff
    #    toc = time.time()

    #    depth_losses.append(depth_recon_loss)
    #    sn_losses.append(sn_recon_loss)
    #    mask_losses.append(mask_cls_loss)

    #    #log_string('Iteration: {} time {}, loss: {}, depth_recon_loss: {}, sn_recon_loss {}, mask_cls_loss {}'.format(i, \
    #    #    toc-tic, loss, depth_recon_loss, sn_recon_loss, mask_cls_loss))

    #    #test_idx += 1

    #log_string(tf_util.toGreen('===========Done testing==========='))
    #toc = time.time()
    #mean_depth_loss = np.mean(np.asarray(depth_losses))
    #mean_sn_loss = np.mean(np.asarray(sn_losses))
    #mean_mask_loss = np.mean(np.asarray(mask_losses))
    #log_string(tf_util.toRed('Test time {}s, depth recon loss: {}, sn recon loss: {}, mask cls loss:{}.'.format(\
    #    toc-tic, mean_depth_loss, mean_sn_loss, mean_mask_loss)))

    #losses = {'loss_depth_recon': mean_depth_loss,\
    #    'loss_sn_recon': mean_sn_loss,\
    #    'loss_mask_cls': mean_mask_loss}

    #return losses
                

            #if i%FLAGS.vis_every_step == 0:
            #    v.process(vis, 'train', i)
            
            #if i > 1000:
            #    break
    #except tf.errors.OutOfRangeError:
    #    print('Done testing')
    #finally:
    #    pass
        #ae.coord.request_stop()


#def get_degree_error(tws0, tws1):
#    error_list = []
#    for i in range(tws0.shape[0]):
#        R0 = tw2R(tws0[i])
#        R = tw2R(tws1[i])
#        delta_R = np.dot(R, R0.T)
#        delta_degree = np.rad2deg(np.linalg.norm(R2tw(delta_R)))
#        error_list.append(delta_degree)
#    return error_list
         
if __name__ == "__main__":
    #MODEL = importlib.import_module(FLAGS.model_file) # import network module
    #MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model_file+'.py')
    
    FLAGS.LOG_DIR = FLAGS.LOG_DIR + '/' + FLAGS.task_name
    #FLAGS.CHECKPOINT_DIR = os.path.join(FLAGS.CHECKPOINT_DIR, FLAGS.task_name)
    #tf_util.mkdir(FLAGS.CHECKPOINT_DIR)
    if not os.path.exists(FLAGS.LOG_DIR):
        print tf_util.toRed('===== %s not exists!.'%FLAGS.LOG_DIR)
        sys.exit()
    #else:
    #    # os.system('rm -rf %s/*'%FLAGS.LOG_DIR)
    #    if not(FLAGS.restore):
    #        
    #        def check_delete():
    #            if FLAGS.force_delete:
    #                return True
    #            delete_key = raw_input(tf_util.toRed('===== %s exists. Delete? [y (or enter)/N] '%FLAGS.LOG_DIR))
    #            return delete_key == 'y' or delete_key == ''
    #        
    #        if check_delete():
    #            os.system('rm -rf %s/*'%FLAGS.LOG_DIR)
    #            #os.system('rm -rf %s/*'%FLAGS.CHECKPOINT_DIR)
    #            print tf_util.toRed('Deleted.'+FLAGS.LOG_DIR)
    #        else:
    #            print tf_util.toRed('Overwrite.')
    #    else:
    #        print tf_util.toRed('To Be Restored...')

    #tf_util.mkdir(os.path.join(FLAGS.LOG_DIR, 'saved_images'))
    #os.system('cp %s %s' % (MODEL_FILE, FLAGS.LOG_DIR)) # bkp of model def
    #os.system('cp train.py %s' % (FLAGS.LOG_DIR)) # bkp of train procedure

    #prepare_plot()
    log_string(tf_util.toYellow('<<<<'+FLAGS.task_name+'>>>> '+str(tf.flags.FLAGS.__flags)))

    ae = AE_rgb2d(FLAGS)
    restore(ae)
    test(ae, FLAGS.test_list_path)

    # z_list = []
    # test_demo_render_z(ae, z_list)

    #FLAGS.LOG_FOUT.close()
