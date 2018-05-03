import os
import sys
import json
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lsm.mvnet import MVNet
from lsm.utils import Bunch, get_session_config
from lsm.models import grid_nets, im_nets, model_vlsm
from lsm.ops import conv_rnns
from models.active_mvnet import MVInputs, SingleInput, SingleInputFactory
from threading import Thread
from Queue import Queue
from time import sleep

sys.path.append(os.path.join('utils'))
from util import downsample
import binvox_rw

np.random.seed(2048)

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
    "03797390": "mug",
    "0000": "combine"
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

        self.input_factory = SingleInputFactory(self)
        
        #from lsm.net_vlsm import net
        #self.vlsm = net
        #log_dir = os.path.join('lsm/models_lsm_v1/vlsm-release/train')
        #vars_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MVNet')
        #ckpt = 'mvnet-100000'
        #self.restorer = tf.train.Saver(var_list=vars_restore)
        #self.restorer.restore(self.sess, os.path.join(log_dir, ckpt))
        #self.get_vlsm()
        #self.get_vlsm_mini()

        if self.FLAGS.GBL_thread:
            self.start_GBL_threads(FLAGS.batch_size)

    def get_vlsm(self):
        #SAMPLE_DIR = os.path.join('data', 'shapenet_sample')
        #im_dir = os.path.join(SAMPLE_DIR, 'renders')
        log_dir = os.path.join('lsm/models_lsm_v1/vlsm-release/train')
        with open(os.path.join(log_dir, 'args.json'), 'r') as f:
            args = json.load(f)
            args = Bunch(args)
        
        bs, ims_per_model = 1, self.max_episode_length
        
        ckpt = 'mvnet-100000'
        net = MVNet(vmin=-0.5, vmax=0.5, vox_bs=bs,
            im_bs=ims_per_model, grid_size=args.nvox,
            im_h=args.im_h, im_w=args.im_w,
            norm=args.norm, mode="TEST")
        
        self.net = model_vlsm(net, im_nets[args.im_net], grid_nets[args.grid_net], conv_rnns[args.rnn])
        vars_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MVNet')
        saver = tf.train.Saver(var_list=vars_restore)
        saver.restore(self.sess, os.path.join(log_dir, ckpt))
    
    #def get_vlsm_mini(self):
    #    #SAMPLE_DIR = os.path.join('data', 'shapenet_sample')
    #    #im_dir = os.path.join(SAMPLE_DIR, 'renders')
    #    log_dir = os.path.join('lsm/models_lsm_v1/vlsm-release/train')
    #    with open(os.path.join(log_dir, 'args.json'), 'r') as f:
    #        args = json.load(f)
    #        args = Bunch(args)
    #    
    #    bs, ims_per_model = 1, self.max_episode_length
    #    
    #    ckpt = 'mvnet-100000'
    #    net = MVNet(vmin=-0.5, vmax=0.5, vox_bs=bs,
    #        im_bs=ims_per_model, grid_size=args.nvox,
    #        im_h=args.im_h, im_w=args.im_w,
    #        norm=args.norm, mode="TEST")
    #    
    #    self.net_mini = model_vlsm(net, im_nets[args.im_net], grid_nets[args.grid_net], conv_rnns[args.rnn],
    #        scope_name='MVNet_mini')
    #    vars_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MVNet_mini')
    #    saver = tf.train.Saver(var_list=vars_restore)
    #    saver.restore(self.sess, os.path.join(log_dir, ckpt))

    def append(self, data_list):

        # data_list: azim/elev list(0:max_episode_length), model_id
        if self.count < self.mem_length:
            self.mem_list.append(data_list)
        else:
            self.mem_list[self.count%self.mem_length] = data_list
        
        self.count += 1

    def read_bv(self, fn, transpose = True):
        with open(fn, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        data = np.float32(model.data)
        if transpose:
            data = np.transpose(data, (0,2,1))
        return data

    def read_vox(self, vox_name, transpose = False):
        vox_model = self.read_bv(vox_name, transpose) 
        vox_factor = self.voxel_resolution * 1.0 / 128
        #vox_model_zoom = ndimg.zoom(vox_model, vox_factor, order=0) # nearest neighbor interpolation
        vox_model_zoom = downsample(vox_model, int(1/vox_factor))

        return vox_model_zoom

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
        return (new_img/255.0).astype(np.float32), mask[..., 0]/255.0

    def read_invZ(self, azim, elev, model_id, resize = True):
        invZ_name = 'invZ_{}_{}.npy'.format(int(azim), int(elev))
        invZ_path = os.path.join(self.data_dir, model_id, invZ_name)
        invZ = np.load(invZ_path)
        if resize:
            invZ = sm.imresize(invZ, (self.FLAGS.resolution, self.FLAGS.resolution), mode = 'F', interp='nearest')
        return invZ

    def get_R(self, azim, elev):
        azim_R = np.zeros((3,3), dtype=np.float32)
        elev_R = np.zeros((3,3), dtype=np.float32)
        azim_rad = np.deg2rad(azim)
        elev_rad = np.deg2rad(elev)
        azim_R[2, 2] = 1
        azim_R[0, 0] = np.cos(azim_rad)
        azim_R[0, 1] = -np.sin(azim_rad)
        azim_R[1, 0] = np.sin(azim_rad)
        azim_R[1, 1] = np.cos(azim_rad)
        elev_R[1, 1] = 1
        elev_R[0, 0] = np.cos(elev_rad)
        elev_R[0, 2] = np.sin(elev_rad)
        elev_R[2, 0] = -np.sin(elev_rad)
        elev_R[2, 2] = np.cos(elev_rad)
        swap_m = np.asarray([[0, 1, 0], [0, 0, 1], [-1, 0, 0]])
        R = np.zeros((3,4), dtype=np.float32)
        R[:, 0:3] = np.matmul(elev_R.T, azim_R.T)
        R[:, 2] = -R[:, 2]
        R[:, 0:3] = np.matmul(swap_m, R[:, 0:3])
        R[0, 3] = 2*np.cos(elev_rad)*np.cos(azim_rad)
        R[1, 3] = 2*np.cos(elev_rad)*np.sin(azim_rad)
        R[2, 3] = 2*np.sin(elev_rad)

        R[:, 3] = -np.matmul(R[0:3, 0:3], R[:, 3])
        #R[2, 3] = 2

        return R 

    def calu_reward(self, vox_curr_batch, vox_next_batch, vox_gt_batch):
        batch_size = vox_gt_batch.shape[0]
        #print 'calu r batch_size: {}'.format(batch_size)
        reward_batch = np.ones((batch_size, ), dtype=np.float32)

        #print vox_curr_batch.shape
        #print vox_next_batch.shape
        #print vox_gt_batch.shape

        if self.FLAGS.reward_type == 'oU':
            calu_r_func = self.calu_IoU_reward
        elif self.FLAGS.reward_type == 'IG':
            calu_r_func = self.calu_IG_reward

        for i in range(batch_size):
            reward_batch[i] = calu_r_func(vox_curr_batch[i], vox_next_batch[i], vox_gt_batch[i])

        return reward_batch

    def calu_IoU_reward(self, vox_curr, vox_next, vox_gt):
        IoU_curr = self.calu_IoU(vox_curr, vox_gt)
        IoU_next = self.calu_IoU(vox_next, vox_gt)

        return (IoU_next - IoU_curr)*100
    
    def calu_IoU(self, a, b, thres=0.5):
        ## do threshold filtering as there are interpolated values
        aa = np.copy(a)
        bb = np.copy(b)
        aa[a > thres] = 1
        aa[a <= thres] = 0
        bb[b > thres] = 1
        bb[b <= thres] = 0

        inter = aa*bb
        sum_inter = np.sum(inter[:])
        union = aa + bb
        union[union > 0.5] = 1
        sum_union = np.sum(union[:])
        return sum_inter*1.0/sum_union

    def calu_IG_reward(self, vox_curr, vox_next, vox_gt):

        cross_entropy_curr = self.calu_cross_entropy(vox_curr, vox_gt)
        cross_entropy_next = self.calu_cross_entropy(vox_next, vox_gt)

        def sigmoid(a):
            return 1.0 / (1 + np.exp(-a))

        return (cross_entropy_next - cross_entropy_curr)*1e-5
        
    def calu_cross_entropy(self, a, b):
        a[np.argwhere(b == 0)] = 1 - a[np.argwhere(b == 0)] 
        a[np.argwhere(a == 0)] += 1e-5

        cross_entropy = np.log(a)
        return np.sum(cross_entropy[:])

    # def get_batch(self, batch_size=32):
        
    #     RGB_list_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution, 3), dtype=np.float32)
    #     invZ_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution, ), dtype=np.float32)
    #     mask_list_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution,), dtype=np.float32)
    #     sn_batch = np.zeros((batch_size, self.max_episode_length, self.resolution, self.resolution, 3), dtype=np.float32)
    #     vox_gt_batch = np.ones((batch_size, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
    #         dtype=np.float32)
    #     #vox_current_batch = np.ones((self.max_episode_length, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
    #     #    dtype=np.float32)
    #     azim_batch = np.zeros((batch_size, self.max_episode_length,), dtype=np.float32)
    #     elev_batch = np.zeros((batch_size, self.max_episode_length,), dtype=np.float32)
    #     actions_batch = np.zeros((batch_size, self.max_episode_length-1,), dtype=np.float32)
    #     R_batch = np.zeros((batch_size, self.max_episode_length, 3, 4), dtype=np.float32)
    #     K_single = np.asarray([[420.0, 0.0, 112.0], [0.0, 420.0, 112.0], [0.0, 0.0, 1]])
    #     K_batch = np.tile(K_single[None, None, ...], (batch_size, self.max_episode_length, 1, 1))  
    #     #K_batch = np.zeros((self.max_episode_length, batch_size, 3, 3), dtype=np.float32)
    #     #state_mask_current = np.ones_like(RGB_list_batch, dtype=np.float32)
    #     #state_mask_next = np.ones_like(RGB_list_batch, dtype=np.float32)
    #     current_idx_list = np.zeros((batch_size,), dtype=np.uint8)
    #     reward_list_batch = np.zeros((batch_size, self.max_episode_length-1,), dtype=np.float32)

    #     for b_idx in range(batch_size):
    #         higher_bound = min(self.count, self.mem_length)
    #         rand_idx = np.random.randint(0, higher_bound)
    #         data_ = self.mem_list[rand_idx]
    #         #print data_.states[0]

    #         azim_batch[b_idx, ...] = np.asarray(data_.states[0])
    #         elev_batch[b_idx, ...] = np.asarray(data_.states[1])
    #         actions_batch[b_idx, ...] = np.asarray(data_.actions)
    #         reward_list_batch[b_idx, ...] = np.asarray(data_.rewards)
    #         model_id = data_.model_id
    #         current_idx = np.random.randint(0, self.max_episode_length-1)
    #         current_idx_list[b_idx] = current_idx
    #         #state_mask_current[b_idx, 0:current_idx, ...] = 0.0
    #         #state_mask_next[b_idx, 0:current_idx+1, ...] = 0.0
    #         voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(self.FLAGS.category, model_id))
    #         vox_gt_batch[b_idx, ...] = self.read_vox(voxel_name)

    #         for l_idx in range(self.max_episode_length):
    #             RGB_list_batch[b_idx, l_idx, ...], mask_list_batch[b_idx, l_idx, ...] = self.read_png_to_uint8(
    #                 azim_batch[b_idx, l_idx], elev_batch[b_idx, l_idx], model_id)

    #             invZ_batch[b_idx, l_idx, ...] = self.read_invZ(azim_batch[b_idx, l_idx],
    #                 elev_batch[b_idx, l_idx], model_id)

    #             R_batch[b_idx, l_idx, ...] = self.get_R(azim_batch[b_idx, l_idx], elev_batch[b_idx, l_idx])

    #             ## TODO: update sn_batch and vox_gt_batch

    #     #print 'K_batch: {}'.format(K_batch.shape)
    #     #feed_dict = {self.net.K: K_batch, self.net.Rcam: R_batch, self.net.ims: RGB_list_batch*state_mask_current}
    #     #pred_voxels = self.sess.run(self.net.prob_vox, feed_dict=feed_dict)
    #     #vox_current_batch = pred_voxels
        
    #     pred_voxels = [] 
    #     for K_single, R_single, RGB_single in zip(K_batch, R_batch, RGB_list_batch):
    #         feed_dict = {self.net.K: K_single[None, ...], self.net.Rcam: R_single[None, ...], self.net.ims:
    #             RGB_single[None, ...]}
    #         pred_voxels_temp = self.sess.run(self.net.prob_vox, feed_dict=feed_dict)
    #         pred_voxels.append(pred_voxels_temp)
    #     pred_voxels = np.squeeze(np.asarray(pred_voxels))
    #     #vox_curr_batch = np.squeeze(pred_voxels[range(batch_size), current_idx_list, ...], axis=-1)
    #     vox_curr_batch = pred_voxels[range(batch_size), current_idx_list, ...]
    #     vox_next_batch = pred_voxels[range(batch_size), current_idx_list+1, ...]

    #     RGB_batch = np.asarray([RGB_list_batch[bi, li, ...] for bi, li in zip(range(batch_size), current_idx_list)],
    #         dtype=np.float32)

    #     ## instance reward
    #     #reward_batch = self.calu_reward(vox_curr_batch, vox_next_batch, vox_gt_batch)
    #     ## cumulated reward
    #     reward_batch = self.get_decay_reward(reward_list_batch, current_idx_list)
    #     action_response_batch = actions_batch[range(batch_size), current_idx_list]

    #     #return RGB_batch, invZ_batch, mask_batch, sn_batch, vox_gt_batch, azim_batch, elev_batch, actions_batch
    #     return RGB_batch, vox_curr_batch, reward_batch, action_response_batch

    def upper_bound(self):
        return min(self.count, self.mem_length)

    def enable_gbl(self):
        if self.FLAGS.GBL_thread:
            self.enabled = True
            self.stopped = False

    def disable_gbl(self):
        if self.FLAGS.GBL_thread:
            self.enabled = False
            while not self.stopped:
                sleep(0.01)
    
    def start_GBL_threads(self, bs):
        self.max_buffer_size = 32
        self.gblbs = bs
        self.gblq = Queue()
        
        self.enabled = True
        self.stopped = False
        
        num_threads = 1

        for i in range(num_threads):
            th = Thread(target = self.loop_gbl)
            th.start()

    def loop_gbl(self):
        while 1:
            if not self.enabled:
                self.stopped = True                
                sleep(0.01)
                continue
            
            if self.gblq.qsize() > self.max_buffer_size:
                sleep(0.01)
                continue

            if self.upper_bound() <= 0:
                sleep(0.01)
                continue #wait for burnin to complete

            item = self._get_batch_list(self.gblbs)
            self.gblq.put(item)
        
    def _get_batch_list(self, batch_size=4):
        mvinputs = MVInputs(self.FLAGS, batch_size = batch_size)
        
        #batch_idx = np.random.choice(self.upper_bound(), batch_size, replace=False)
        for b_idx in range(batch_size):
            rand_idx = np.random.randint(0, self.upper_bound())
            data_ = self.mem_list[rand_idx]

            azimuths = np.asarray(data_.states[0])
            elevations = np.asarray(data_.states[1])
            actions = np.asarray(np.expand_dims(data_.actions, axis=1))
            penalties = np.abs(azimuths-azimuths[0]) + np.abs(elevations-elevations[0])

            model_id = data_.model_id
            voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(self.FLAGS.category, model_id))
            voxel = self.read_vox(voxel_name)

            mvinputs.put_voxel(voxel, batch_idx = b_idx)

            for l_idx in range(self.max_episode_length):
                
                azimuth = azimuths[l_idx]
                elevation = elevations[l_idx]
                action = actions[l_idx] if (l_idx < self.max_episode_length-1) else None
                penalty = penalties[l_idx]

                single_input = self.input_factory.make(azimuth, elevation, model_id, action = action, penalty = penalty)
                mvinputs.put(single_input, episode_idx = l_idx, batch_idx = b_idx)

        return mvinputs

    def get_batch_list(self, batch_size = 4):
        if self.FLAGS.GBL_thread:
            assert batch_size == self.gblbs
            return self.gblq.get()
        else:
            return self._get_batch_list(batch_size)

    def get_batch_list_random(self, env, batch_size=4):
        mvinputs = MVInputs(self.FLAGS, batch_size = batch_size)
        rand_idxes = np.random.randint(0, env.train_len, (batch_size,))
        azimuths_batch = np.random.choice(env.azim_all, (batch_size, self.max_episode_length))
        elevations_batch = np.random.choice(env.elev_all, (batch_size, self.max_episode_length))  
        for b_idx, (rand_idx, azimuths, elevations) in enumerate(zip(rand_idxes, azimuths_batch, elevations_batch)):
            model_id = env.trainval_list[rand_idx] 
            voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(self.FLAGS.category, model_id))
            voxel = self.read_vox(voxel_name)
            penalties = np.abs(azimuths-azimuths[0]) + np.abs(elevations-elevations[0])

            for l_idx in range(self.max_episode_length):
                azimuth = azimuths[l_idx]
                elevation = elevations[l_idx]
                action = None
                penalty = penalties[l_idx]

                single_input = self.input_factory.make(azimuth, elevation, model_id, action = action, penalty = penalty)
                mvinputs.put(single_input, episode_idx = l_idx, batch_idx = b_idx)
        return mvinputs
        #for b_idx in range(batch_size):
        #    rand_idx = np.random.randint(0, self.upper_bound())
        #    data_ = self.mem_list[rand_idx]

        #    azimuths = np.asarray(data_.states[0])
        #    elevations = np.asarray(data_.states[1])
        #    actions = np.asarray(np.expand_dims(data_.actions, axis=1))
        #    penalties = np.abs(azimuths-azimuths[0]) + np.abs(elevations-elevations[0])

        #    model_id = data_.model_id
        #    voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(self.FLAGS.category, model_id))
        #    voxel = self.read_vox(voxel_name)

        #    mvinputs.put_voxel(voxel, batch_idx = b_idx)

        #    for l_idx in range(self.max_episode_length):
        #        
        #        azimuth = azimuths[l_idx]
        #        elevation = elevations[l_idx]
        #        action = actions[l_idx] if (l_idx < self.max_episode_length-1) else None
        #        penalty = penalties[l_idx]

        #        single_input = self.input_factory.make(azimuth, elevation, model_id, action = action, penalty = penalty)
        #        mvinputs.put(single_input, episode_idx = l_idx, batch_idx = b_idx)

        #return mvinputs
    
    def get_decay_reward(self, reward_list_batch, current_idx_list):
        gamma = self.FLAGS.gamma

        reward_batch = np.zeros((self.FLAGS.batch_size,), dtype=np.float32)
        for b_idx in range(self.FLAGS.batch_size):
            for l_i in range(self.max_episode_length-2, current_idx_list[b_idx]-2, -1):
                reward_batch[b_idx] += gamma*reward_batch[b_idx] + reward_list_batch[b_idx, l_i]

        return reward_batch

    def get_vox_pred(self, RGB_list, R_list, K_list, seq_idx):
        feed_dict = {self.net.K: K_list, self.net.Rcam: R_list[None, ...], 
            self.net.ims: RGB_list[None, ...]}
        pred_voxels = self.sess.run(self.net.prob_vox, feed_dict=feed_dict)

        return pred_voxels[0, ...]
        
    def get_seq_rewards(self, RGB_list, R_list, K_list, model_id):
            
        voxel_name = os.path.join('voxels', '{}/{}/model.binvox'.format(self.FLAGS.category, model_id))
        vox_gt = self.read_vox(voxel_name)
        feed_dict = {self.net.K: K_list, self.net.Rcam: R_list[None, ...], 
            self.net.ims: RGB_list[None, ...]}
        pred_voxels = self.sess.run(self.net.prob_vox, feed_dict=feed_dict)
        rewards = []
        pred_voxels = np.squeeze(pred_voxels[0])
        for i in range(1, pred_voxels.shape[0]):
            r = self.calu_reward(pred_voxels[None, i-1, ...], pred_voxels[None, i, ...], vox_gt[None, ...])
            rewards.append(r[0])

        ### extract mean
        #rewards = np.asarray(rewards)
        #rewards -= np.mean(rewards)

        return rewards

class trajectData():
    def __init__(self, states, actions, model_id):
        self.states = states
        self.actions = actions
        self.model_id = model_id
