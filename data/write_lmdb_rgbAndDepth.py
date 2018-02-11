# scp jerrypiglet@128.237.133.169:Bitsync/3dv2017_PBA/data/write_lmdb_imageAndShape_direct2.py . && CUDA_VISIBLE_DEVICES=3 python write_lmdb_imageAndShape_direct2.py --ae_file '/newfoundland/rz1/log/finalAE_1e-5_bnNObn_car24576__bb10/'

import numpy as np
import math
import os, sys
import os.path
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import *
import gc
#from plyfile import PlyData, PlyElement
import tables
from multiprocessing import Pool
import os
# from contextlib import closingy
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))

#global FLAGS
#flags = tf.flags
#flags.DEFINE_string('ae_file', '', '')
#flags.DEFINE_boolean('if_en_bn', True, 'If use batch normalization for the mesh decoder')
#flags.DEFINE_boolean('if_gen_bn', False, 'If use batch normalization for the mesh generator')
#flags.DEFINE_float('bn_decay', 0.9, 'Decay rate for batch normalization [default: 0.9]')
#FLAGS = flags.FLAGS

sample_num = 24576
resolution = 128
VIEWS = 200
#BASE_OUT_DIR = '/home/rz1/Documents/Research/3dv2017_PBA_out/'
BASE_OUT_DIR = './data_cache'
# pcd_path = '/home/rz1/Documents/Research/3dv2017_PBA_out/PCDs/'

categories = [
    # "02691156", #airplane
    # "02828884",
    # "02933112",
    "03001627", #chair
    # "03211117",
    # "03636649",
    # "03691459",
    # "04090263",
    # "04256520",
    # "04379243",
    # "04401088",
    # "04530566",
    # "02958343" #car
]
cat_name = {
    "02691156" : "airplane",
    # "02828884",
    # "02933112",
    "03001627" : "chair",
    # "03211117",
    # "03636649",
    # "03691459",
    # "04090263",
    # "04256520",
    # "04379243",
    # "04401088",
    # "04530566",
    "02958343" : "car"
}

azim_all = np.linspace(0, 360, 9)
azim_all = azim_all[0:-1]
elev_all = np.linspace(-30, 30, 5)

class lmdb_writer(DataFlow):
    def __init__(self, model_ids):
        self.model_ids = model_ids

    def get_data(self):
        for model_id in self.model_ids:
            #ply_name = pcd_path + '%s/%s_%d.ply'%(category_name, model_id, sample_num)
            mat_name = render_out_path + '/%s_tw.mat'%model_id
            #try:
            #    #plydata = PlyData.read(ply_name)
            #    mat_struct = sio.loadmat(mat_name)
            #    #gc.collect()
            #except ValueError:
            #    print '++++++ Oops! Discrading model %s'%model_id
            #    continue
            #pcd = np.concatenate((np.expand_dims(plydata['vertex']['x'], 1), np.expand_dims(plydata['vertex']['z'], 1), np.expand_dims(plydata['vertex']['y'], 1)), 1)
            #pcd = np.asarray(pcd, dtype='float32')
            #angles = mat_struct['angles_list'] #views*4; unit axis; angle in radians
            #tws = mat_struct['tws'] #views*3
            #angles = mat_struct['angles'] #views*3; in degrees

            for a in azim_all:
                for e in elev_all:
                    image_name = os.path.join(render_out_path, '{}/RGB_{}_{}.png'.format(model_id, int(a), int(e)))
                    invZ_name = os.path.join(render_out_path, '{}/invZ_{}_{}.npy'.format(model_id, int(a), int(e)))
                    rgb_single, mask_single = read_png_to_uint8(image_name)
                    invZ_single = np.load(invZ_name)
                    print mask_single.shape
                    print mask_single.dtype
                    if invZ_single.ndim == 2:
                        invZ_single = invZ_single[:, :, None]
                    if mask_single.ndim == 2:
                        mask_single = mask_single[:, :, None]
                    if mask_single.shape[2] == 3:
                        mask_single = mask_single[:, :, 0]
                        mask_single = mask_single[:, :, None]

                    yield [rgb_single, invZ_single, mask_single, np.asarray([a, e, 0.], dtype=np.float32)]

            #for view in range(VIEWS):
            #    image_name = render_out_path + '/%s/%d_0.png'%(model_id, view)
            #    rgb_single = read_png_to_uint8(image_name)
            #    axis_angle_single = np.asarray(axis_angles[view], dtype='float32')
            #    tw_single = np.asarray(tws[view], dtype='float32')
            #    angle_single = np.asarray(angles[view], dtype='float32')
            #    # print self.features_dict[model_id]
            #    yield [pcd, axis_angle_single, tw_single, angle_single, rgb_single, self.features_dict[model_id]]
                
    def size(self):
        return len(self.model_ids) * VIEWS
    
def get_surface_from_depth(invZ):
    pass

def read_png_to_uint8(img_name):
    img = mpimg.imread(img_name)
    new_img = img[:, :, :3]
    mask = img[:, :, 3]
    mask = np.tile(np.expand_dims(mask, 2), (1, 1,3))
    new_img = new_img * mask + np.ones_like(new_img, dtype=np.float32) * (1.0 - mask)
    return (new_img*255.).astype(np.uint8), mask

def get_models(category_name, splits = ['train', 'test', 'val']):
    model_ids = []
    for split in splits:
        listFile = "./render_scripts/lists/PTNlist_v2/%s_%sids.txt"%(category_name, split)
        listFile = os.path.join("./render_scripts/lists/{}_debug.txt".format(category_name))
        listFile = os.path.join("./render_scripts/lists/{}_lists/{}_idx.txt".format(category_name, split))
        #print listFile
        #sys.exit()
        with open(listFile) as file:
            for line in file:
                model = line.strip()
                model_ids.append(model)
                #mat_name = render_out_path + '/%s_tw.mat'%model
                #if os.path.isfile(mat_name):
                #    model_ids.append(model)
                #    # print 'added mat for %s to in queue.'%model
                #    # print os.path.isfile(mat_name)
                #else:
                #    print 'mat for %s does not exist; skipped.'%model
                # print mat_name
    model_ids.sort()
    modelN = len(model_ids)
    print '+++ Working on category %s with %d models in total...'%(category_name, modelN)
    model_ids = shuffle(model_ids, random_state=0)
    return model_ids

#def lrelu(x, leak=0.2, name="lrelu"):
#    with tf.variable_scope(name):
#        f1 = 0.5 * (1 + leak)
#        f2 = 0.5 * (1 - leak)
#        return f1 * x + f2 * abs(x)
        
#class PCD_ae_mini(object):
#    def __init__(self, FLAGS):
#        self.FLAGS = FLAGS
#        self.activation_fn = lrelu
#        self.batch_size = 1
#        self.num_point = sample_num
#        self.is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training_pl')
#
#        self._create_network()
#        
#        restore_vars = slim.get_variables_to_restore(include=["encoder"])
#        print [op.name for op in restore_vars]
#        self.restorer_ae = tf.train.Saver(slim.get_variables_to_restore(include=["encoder", "generator"]))
#            
#        # Create a session
#        config = tf.ConfigProto()
#        config.gpu_options.allow_growth = True
#        config.allow_soft_placement = True
#        config.log_device_placement = False
#        self.sess = tf.Session(config=config)
#        self.sess.run(tf.global_variables_initializer())
#
#    def _create_encoder(self, input_sample, trainable=False, if_bn=True, reuse=False, scope_name='encoder'):
#         with tf.variable_scope(scope_name) as scope:
#            if reuse:
#                scope.reuse_variables()
#
#            if if_bn:
#                print '=== Using BN for ENCODER!'
#                batch_normalizer_en = slim.batch_norm
#                batch_norm_params_en = {'is_training': self.is_training_pl, 'decay': self.FLAGS.bn_decay}
#            else:
#                print '=== NOT Using BN for ENCODER!'
#                batch_normalizer_en = None
#                batch_norm_params_en = None
#
#            with slim.arg_scope([slim.fully_connected, slim.conv2d], 
#                    activation_fn=self.activation_fn,
#                    trainable=trainable,
#                    normalizer_fn=batch_normalizer_en,
#                    normalizer_params=batch_norm_params_en):
#                net = slim.conv2d(input_sample, 64, kernel_size=[1,3], stride=[1,1], padding='VALID',scope='conv1')
#                net = slim.conv2d(net, 64, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv2')
#
#                net = slim.conv2d(net, 64, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv3')
#                net = slim.conv2d(net, 128, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv4')
#                net = slim.conv2d(net, 1024, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv5')
#                feat = slim.max_pool2d(net, [self.num_point,1], padding='VALID', scope='maxpool')
#                
#
#            feat_before_VAE = tf.reshape(feat, [-1, 1024]) #[32, 1024]
#            feat = feat_before_VAE
#
#            return feat
#
#    def _create_generator(self, feat, trainable=False, if_bn=False, reuse=False, scope_name='generator'):
#         with tf.variable_scope(scope_name) as scope:
#            if reuse:
#                scope.reuse_variables()
#
#            if if_bn:
#                print '=== Using BN for GENERATOR!'
#                batch_normalizer_gen = slim.batch_norm
#                batch_norm_params_gen = {'is_training': self.is_training_pl, 'decay': self.FLAGS.bn_decay}
#            else:
#                print '=== NOT Using BN for GENERATOR!'
#                batch_normalizer_gen = None
#                batch_norm_params_gen = None
#            weights_regularizer = None
#
#            with slim.arg_scope([slim.fully_connected], 
#                    activation_fn=self.activation_fn,
#                    trainable=trainable,
#                    normalizer_fn=batch_normalizer_gen,
#                    normalizer_params=batch_norm_params_gen, 
#                    weights_regularizer=weights_regularizer):
#                x_additional = slim.fully_connected(feat, 2048, scope='gen_fc1')
#                x_additional = slim.fully_connected(x_additional, 4096, scope='gen_fc2')
#                x_additional = slim.fully_connected(x_additional, 8192, scope='gen_fc3')
#                x_additional = slim.fully_connected(x_additional, 8192*3, scope='gen_fc4',
#                    activation_fn=None, normalizer_fn=None, normalizer_params=None)
#            
#            x_recon=tf.reshape(x_additional,(-1,8192,3))
#            
#            x_additional_conv = tf.reshape(feat, [-1, 4, 4, 64])
#            with slim.arg_scope([slim.conv2d_transpose], 
#                    activation_fn=self.activation_fn,
#                    trainable=trainable,
#                    normalizer_fn=batch_normalizer_gen,
#                    normalizer_params=batch_norm_params_gen, 
#                    weights_regularizer=weights_regularizer):
#                gen_deconv1 = slim.conv2d_transpose(x_additional_conv, 256, kernel_size=[3,3], stride=[1,1], padding='VALID',scope='gen_deconv1')
#                gen_deconv2 = slim.conv2d_transpose(gen_deconv1, 128, kernel_size=[3,3], stride=[1,1], padding='VALID',scope='gen_deconv2')
#                gen_deconv3 = slim.conv2d_transpose(gen_deconv2, 64, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv3')
#                gen_deconv4 = slim.conv2d_transpose(gen_deconv3, 64, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv4')
#                gen_deconv5 = slim.conv2d_transpose(gen_deconv4, 32, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv6')
#                gen_deconv6 = slim.conv2d_transpose(gen_deconv5, 3, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv7',
#                    activation_fn=None, normalizer_fn=None, normalizer_params=None)
#
#            x_recon_conv = tf.reshape(gen_deconv6, [-1, 128*128, 3])
#            x_recon = tf.concat([x_recon, x_recon_conv],1)
#
#            return x_recon
#
#    def _create_network(self):
#        self.point_cloud_pl = tf.placeholder(tf.float32, shape=[1, sample_num, 3, 1])
#        self.style_gnd = self._create_encoder(self.point_cloud_pl, trainable=False, if_bn=self.FLAGS.if_en_bn, reuse=False, scope_name='encoder')
#        self.point_cloud_recon = self._create_generator(self.style_gnd, trainable=False, if_bn=self.FLAGS.if_gen_bn, reuse=False, scope_name='generator')
#
#def get_features(model_ids, ae):
#    features_dict = {}
#    for idx, model_id in enumerate(model_ids):
#        ply_name = pcd_path + '%s/%s_%d.ply'%(category_name, model_id, sample_num)
#        try:
#            plydata = PlyData.read(ply_name)
#            gc.collect()
#        except ValueError:
#            continue
#        pcd = np.concatenate((np.expand_dims(plydata['vertex']['x'], 1), np.expand_dims(plydata['vertex']['z'], 1), np.expand_dims(plydata['vertex']['y'], 1)), 1)
#        pcd = np.asarray(pcd, dtype='float32')
#        feed_dict = {ae.is_training_pl: False, ae.point_cloud_pl: np.expand_dims(np.expand_dims(pcd, 0), -1)}
#        style_gnd, pcl_recon = ae.sess.run([ae.style_gnd, ae.point_cloud_recon], feed_dict=feed_dict)
#        features_dict[model_id] = style_gnd.reshape([-1])
#        if idx % 1000 == 0:
#            print 'Getting features for model %d/%d...'%(idx, len(model_ids))
#
#        # fig = plt.figure(1)
#        # plt.clf()
#        # plt.subplot(211)
#        # plt.imshow(point_cloud_three_views(pcd))
#        # plt.axis('off')
#        # plt.subplot(212)
#        # plt.imshow(point_cloud_three_views(np.squeeze(pcl_recon)))
#        # plt.axis('off')
#        # fig.canvas.draw()
#        # plt.pause(0.001)
#    print len(features_dict)
#    return features_dict

if __name__ == "__main__":
    #ae = PCD_ae_mini(FLAGS)
    #if "ckpt" not in FLAGS.ae_file:
    #    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.ae_file)
    #else:
    #    latest_checkpoint = FLAGS.ae_file
    #print "-----> AE restoring from: %s..."%latest_checkpoint
    #ae.restorer_ae.restore(ae.sess, latest_checkpoint)
    #print "-----> AE restored."
    
    # splits_list = [['train', 'test', 'val'], ['train', 'val'], ['test']]
    splits_list = [['train', 'val'], ['test']]
    # splits_list = [['test']]
    for category_name in categories:
        render_out_path = os.path.join(BASE_OUT_DIR, 'blender_renderings/%s/res%d_chair_debug_nonorm'%(category_name, \
            resolution))
        # render_out_path = '/newfoundland/rz1/res128_random_randLampbb8'
        for splits in splits_list:        
            if splits == ['train', 'test', 'val']:
                lmdb_name_append = 'all'
            else:
                if splits == ['train', 'val']:
                    lmdb_name_append = 'train'
                else:
                    if splits == ['test']:
                        lmdb_name_append = 'test'
                    else:
                        print '++++++ Oops! Split cases not valid!', splits
                        sys.exit(0)
            # write_path = "/home/rz1/Documents/Research/3dv2017_PBA/data/lmdb"
            # write_path = '/newfoundland/rz1/lmdb'
            write_path = './lmdb'
            # write_path = '/data_tmp/lmdbqqqq'
            lmdb_write = write_path + "/random_randomLamp0822_%s_%d_%s_imageAndShape_single.lmdb"%(cat_name[category_name], sample_num, lmdb_name_append)
            lmdb_write = os.path.join(write_path, 'rgb2depth_single_0209.lmdb')
            lmdb_write = os.path.join(write_path, 'rgb2depth_single_{}.lmdb'.format(split))

            command = 'rm -rf %s'%lmdb_write
            print command
            os.system(command)

            model_ids = get_models(category_name, splits = splits)
            print model_ids[:3]
            print "====== Writing %d models to %s; split: "%(len(model_ids), lmdb_write), splits
            #features_dict = get_features(model_ids, ae)
            ds0 = lmdb_writer(model_ids)
            # ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
            dftools.dump_dataflow_to_lmdb(ds0, lmdb_write)
