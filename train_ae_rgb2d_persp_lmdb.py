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

from ae_rgb2depth import AE_rgb2d


np.random.seed(0)
tf.set_random_seed(0)
global FLAGS
flags = tf.flags
flags.DEFINE_integer('gpu', 0, "GPU to use [default: GPU 0]")
# task and control (yellow)
flags.DEFINE_string('model_file', 'pcd_ae_1_lmdb', 'Model name')
flags.DEFINE_string('cat_name', 'airplane', 'Category name')
#flags.DEFINE_string('LOG_DIR', '/newfoundland/rz1/log/summary', 'Log dir [default: log]')
flags.DEFINE_string('LOG_DIR', './log/summary', 'Log dir [default: log]')
#flags.DEFINE_string('CHECKPOINT_DIR', '/newfoundland/rz1/log', 'Log dir [default: log]')
flags.DEFINE_string('CHECKPOINT_DIR', './log', 'Log dir [default: log]')
flags.DEFINE_string('task_name', 'tmp', 'task name to create under /LOG_DIR/ [default: tmp]')
flags.DEFINE_boolean('restore', False, 'If resume from checkpoint')
flags.DEFINE_string('ae_file', '', '')
# train (green)
flags.DEFINE_integer('num_point', 2048, 'Point Number [256/512/1024/2048] [default: 1024]')
flags.DEFINE_integer('resolution', 128, '')
flags.DEFINE_integer('voxel_resolution', 32, '')
flags.DEFINE_integer('batch_size', 16, 'Batch Size during training [default: 32]')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate [default: 0.001]') #used to be 3e-5
flags.DEFINE_float('momentum', 0.95, 'Initial learning rate [default: 0.9]')
flags.DEFINE_string('optimizer', 'adam', 'adam or momentum [default: adam]')
flags.DEFINE_integer('decay_step', 5000000, 'Decay step for lr decay [default: 200000]')
flags.DEFINE_float('decay_rate', 0.7, 'Decay rate for lr decay [default: 0.8]')
# arch (magenta)
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
flags.DEFINE_integer("test_every_step", 20, "test every ? step")
flags.DEFINE_boolean("if_draw", True, "if draw latent")
flags.DEFINE_integer("draw_every_step", 1000, "draw every ? step")
flags.DEFINE_boolean("if_init_i", False, "if init i from 0")
flags.DEFINE_integer("init_i_to", 1, "init i to")
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
    saved_checkpoint = ae.saver.save(ae.sess, \
        FLAGS.CHECKPOINT_DIR + '/step%d-epoch%d-batch%d.ckpt' % (step, epoch, batch), \
        global_step=step)
    log_string(tf_util.toBlue("-----> Model saved to file: %s; step = %d" % (saved_checkpoint, step)))

def restore(ae):
    pass

def train(ae):
    
    i = 0 
    try:
        while not ae.coord.should_stop():
            ae.sess.run(ae.assign_i_op, feed_dict={ae.set_i_to_pl: i})

            tic = time.time()
            feed_dict = {ae.is_training: True, ae.data_loader.is_training: True}
            
            ops_to_run = [
                ae.optimizer, ae.merge_train, ae.counter, ae.loss,
                ae.depth_recon_loss, ae.sn_recon_loss, ae.mask_cls_loss
            ]

            stuff = ae.sess.run(ops_to_run, feed_dict = feed_dict)
            opt, summary, step, loss, depth_recon_loss, sn_recon_loss, mask_cls_loss = stuff

            log_string('Iteration: {}, loss: {}, depth_recon_loss: {}, sn_recon_loss {}, mask_cls_loss {}'.format(i, \
                loss, depth_recon_loss, sn_recon_loss, mask_cls_loss))

            i += 1

            if i%FLAGS.save_every_step == 0:
                save(ae, i, i, i)

            if i%FLAGS.test_every_step == 0:
                test(ae)

            #if i > 1000:
            #    break
    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        ae.coord.request_stop()

    ae.coord.join(ae.threads)
    ae.sess.close()


def test(ae):
    pass


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
    FLAGS.CHECKPOINT_DIR = os.path.join(FLAGS.CHECKPOINT_DIR, FLAGS.task_name)
    tf_util.mkdir(FLAGS.CHECKPOINT_DIR)
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
                os.system('rm -rf %s/*'%FLAGS.CHECKPOINT_DIR)
                print tf_util.toRed('Deleted.'+FLAGS.LOG_DIR+FLAGS.CHECKPOINT_DIR)
            else:
                print tf_util.toRed('Overwrite.')
        else:
            print tf_util.toRed('To Be Restored...')

    tf_util.mkdir(os.path.join(FLAGS.LOG_DIR, 'saved_images'))
    #os.system('cp %s %s' % (MODEL_FILE, FLAGS.LOG_DIR)) # bkp of model def
    #os.system('cp train.py %s' % (FLAGS.LOG_DIR)) # bkp of train procedure


    FLAGS.LOG_FOUT = open(os.path.join(FLAGS.LOG_DIR, 'log_train.txt'), 'w')
    FLAGS.LOG_FOUT.write(str(FLAGS)+'\n')

    #prepare_plot()
    log_string(tf_util.toYellow('<<<<'+FLAGS.task_name+'>>>> '+str(tf.flags.FLAGS.__flags)))

    ae = AE_rgb2d(FLAGS)
    if FLAGS.restore:
        restore(ae)
    train(ae)

    # z_list = []
    # test_demo_render_z(ae, z_list)

    FLAGS.LOG_FOUT.close()
