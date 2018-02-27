import os
import sys
import json
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lsm.mvnet import MVNet
from lsm.utils import Bunch, get_session_config
from lsm.models import grid_nets, im_nets, model_vlsm
from lsm.ops import conv_rnns

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
    "02958343" : "car",
    "03797390": "mug"
}

class ReplayMemory():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.mem_length = FLAGS.mem_length
        self.count = 0
        self.mem_list = []
        self.max_episode_length = FLAGS.max_episode_length
        self.voxel_resolution = FLAGS.voxel_resolution
        self.resolution = FLAGS.resolution
        self.data_dir = 'data/data_cache/blender_renderings/{}/res128_{}_all/'.format(self.FLAGS.category,
            cat_name[self.FLAGS.category])
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)

        #from lsm.net_vlsm import net
        #self.vlsm = net
        #log_dir = os.path.join('lsm/models_lsm_v1/vlsm-release/train')
        #vars_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MVNet')
        #ckpt = 'mvnet-100000'
        #self.restorer = tf.train.Saver(var_list=vars_restore)
        #self.restorer.restore(self.sess, os.path.join(log_dir, ckpt))
        self.get_vlsm()

    def get_vlsm(self):
        #SAMPLE_DIR = os.path.join('data', 'shapenet_sample')
        #im_dir = os.path.join(SAMPLE_DIR, 'renders')
        log_dir = os.path.join('lsm/models_lsm_v1/vlsm-release/train')
        with open(os.path.join(log_dir, 'args.json'), 'r') as f:
            args = json.load(f)
            args = Bunch(args)
        
        bs, ims_per_model = self.FLAGS.batch_size, self.max_episode_length
        
        ckpt = 'mvnet-100000'
        net = MVNet(vmin=-0.5, vmax=0.5, vox_bs=bs,
            im_bs=ims_per_model, grid_size=args.nvox,
            im_h=args.im_h, im_w=args.im_w,
            norm=args.norm, mode="TEST")
        
        self.net = model_vlsm(net, im_nets[args.im_net], grid_nets[args.grid_net], conv_rnns[args.rnn])
        vars_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MVNet')
        saver = tf.train.Saver(var_list=vars_restore)
        saver.restore(self.sess, os.path.join(log_dir, ckpt))

    def append(self, data_list):
        # data_list: azim/elev list(0:max_episode_length), model_id
        if self.count < self.mem_length:
            self.mem_list.append(data_list)
        else:
            self.mem_list[self.count%self.mem_length] = data_list
        
        self.count += 1

    def read_png_to_uint8(self, azim, elev, model_id):
        img_name = 'RGB_{}_{}.png'.format(int(azim), int(elev))
        img_path = os.path.join(self.data_dir, model_id, img_name)
        img = mpimg.imread(img_path)
        new_img = img[:, :, :3]
        mask = img[:, :, 3]
        mask = np.tile(np.expand_dims(mask, 2), (1, 1, 3))
        new_img = new_img * mask + np.ones_like(new_img, dtype=np.float32) * (1.0 - mask)
        new_img = sm.imresize(new_img, (self.FLAGS.resolution, self.FLAGS.resolution, 3))
        mask = sm.imresize(mask, (self.FLAGS.resolution, self.FLAGS.resolution), interp='nearest') 
        return (new_img*255.).astype(np.uint8), mask[..., 0]

    def read_invZ(self, azim, elev, model_id):
        invZ_name = 'invZ_{}_{}.npy'.format(int(azim), int(elev))
        invZ_path = os.path.join(self.data_dir, model_id, invZ_name)
        invZ = np.load(invZ_path)
        invZ = sm.imresize(invZ, (self.FLAGS.resolution, self.FLAGS.resolution))
        return invZ

    def get_R(self, azim, elev):
        azim_R = np.zeros((3,3), dtype=np.float32)
        elev_R = np.zeros((3,3), dtype=np.float32)
        azim_rad = np.deg2rad(azim)
        elev_rad = np.deg2rad(elev)
        azim_R[1, 1] = np.cos(azim_rad)
        azim_R[1, 2] = -np.sin(azim_rad)
        azim_R[2, 1] = np.sin(azim_rad)
        azim_R[2, 2] = np.cos(azim_rad)
        elev_R[0, 0] = np.cos(elev_rad)
        elev_R[0, 2] = np.sin(elev_rad)
        elev_R[2, 0] = -np.sin(elev_rad)
        elev_R[2, 2] = np.cos(elev_rad)
        R = np.zeros((3,4), dtype=np.float32)
        R[0:3, 0:3] = np.matmul(elev_R, azim_R)
        R[0, 3] = 2*np.cos(elev_rad)*np.cos(azim_rad)
        R[1, 3] = 2*np.cos(elev_rad)*np.sin(azim_rad)
        R[2, 3] = 2*np.sin(elev_rad)

        return np.matmul(elev_R, azim_R) 

    def calu_reward(self, vox_curr_batch, vox_next_batch, vox_gt_batch):
        batch_size = vox_gt_batch.shape[0]

        return np.ones((batch_size, ), dtype=np.float32)

    def get_batch(self, batch_size=32):
        
        RGB_list_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution, 3), dtype=np.float32)
        invZ_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution, ), dtype=np.float32)
        mask_list_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution,), dtype=np.float32)
        sn_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution, 3), dtype=np.float32)
        vox_gt_batch = np.ones((self.max_episode_length, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
            dtype=np.float32)
        vox_current_batch = np.ones((self.max_episode_length, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
            dtype=np.float32)
        azim_batch = np.zeros((batch_size, self.max_episode_length,), dtype=np.float32)
        elev_batch = np.zeros((batch_size, self.max_episode_length,), dtype=np.float32)
        actions_batch = np.zeros((batch_size, self.max_episode_length-1,), dtype=np.float32)
        R_batch = np.zeros((batch_size, self.max_episode_length, 3, 4), dtype=np.float32)
        K_single = np.asarray([[420.0, 0.0, 112.0], [0.0, 420.0, 112.0], [0.0, 0.0, 1]])
        K_batch = np.tile(K_single[None, None, ...], (batch_size, self.max_episode_length, 1, 1))  
        #K_batch = np.zeros((self.max_episode_length, batch_size, 3, 3), dtype=np.float32)
        state_mask_current = np.ones_like(RGB_list_batch, dtype=np.float32)
        state_mask_next = np.ones_like(RGB_list_batch, dtype=np.float32)
        current_idx_list = np.zeros((batch_size,), dtype=np.uint8)

        for b_idx in range(batch_size):
            higher_bound = min(self.count, self.mem_length)
            rand_idx = np.random.randint(0, higher_bound)
            data_ = self.mem_list[rand_idx]

            azim_batch[b_idx, ...] = np.asarray(data_.states[0])
            elev_batch[b_idx, ...] = np.asarray(data_.states[1])
            actions_batch[b_idx, ...] = np.asarray(data_.actions)
            model_id = data_.model_id
            current_idx = np.random.randint(0, self.max_episode_length-1)
            current_idx_list[b_idx] = current_idx
            state_mask_current[b_idx, 0:current_idx, ...] = 0.0
            state_mask_next[b_idx, 0:current_idx+1, ...] = 0.0

            for l_idx in range(self.max_episode_length):
                RGB_list_batch[b_idx, l_idx, ...], mask_list_batch[b_idx, l_idx, ...] = self.read_png_to_uint8(
                    azim_batch[b_idx, l_idx], elev_batch[b_idx, l_idx], data_.model_id)

                invZ_batch[b_idx, l_idx, ...] = self.read_invZ(azim_batch[b_idx, l_idx],
                    elev_batch[b_idx, l_idx], data_.model_id)

                R_batch[b_idx, l_idx, :, 0:3] = self.get_R(azim_batch[b_idx, l_idx], elev_batch[b_idx, l_idx])

                ## TODO: update sn_batch and vox_gt_batch

        feed_dict = {self.net.K: K_batch, self.net.Rcam: R_batch, self.net.ims: RGB_list_batch*state_mask_current}
        pred_voxels = self.sess.run(self.net.prob_vox, feed_dict=feed_dict)
        vox_current_batch = pred_voxels
        
        feed_dict = {self.net.K: K_batch, self.net.Rcam: R_batch, self.net.ims: RGB_list_batch*state_mask_next}
        pred_voxels = self.sess.run(self.net.prob_vox, feed_dict=feed_dict)
        vox_next_batch = pred_voxels

        RGB_batch = np.asarray([RGB_list_batch[bi, li, ...] for bi, li in zip(range(batch_size), current_idx_list)],
            dtype=np.float32)

        reward_batch = self.calu_reward(vox_current_batch, vox_next_batch, vox_gt_batch)

        #return RGB_batch, invZ_batch, mask_batch, sn_batch, vox_gt_batch, azim_batch, elev_batch, actions_batch
        return RGB_batch, vox_current_batch, reward_batch, actions_batch
