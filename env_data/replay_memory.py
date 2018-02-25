import os
import sys
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ReplayMemory():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.mem_length = FLAGS.mem_length
        self.count = 0
        self.mem_list = []
        self.max_episode_length = FLAGS.max_episode_length
        self.voxel_resolution = FLAGS.voxel_resolution
        self.data_dir = 'data/data_cache/blender_renderings/{}/res{}_{}_all/'.format(self.category,
            FLAGS.resolution, cat_name[self.category])
        pass

    def append(self, data_list):
        if self.count < self.mem_length:
            self.mem_list.append(data_list)
        else:
            self.mem_list[self.count%self.mem_length] = data_list
        
        self.count += 1

    def read_png_to_uint8(self, azim, elev, model_id):
        img_name = 'RGB_{}_{}.png'.format(int(azim), int(elev))
        img_path == os.path.join(self.data_dir, model_id, img_name)
        img = mpimg.imread(img_path)
        new_img = img[:, :, :3]
        mask = img[:, :, 3]
        mask = np.tile(np.expand_dims(mask, 2), (1, 1, 3))
        new_img = new_img * mask + np.ones_like(new_img, dtype=np.float32) * (1.0 - mask)
        return (new_img*255.).astype(np.uint8), mask

    def read_invZ(self, azim, elev, model_id):
        invZ_name = 'invZ_{}_{}.npy'.format(int(azim), int(elev))
        invZ_path = os.path.join(self.data_dir, model_id, invZ_name)
        invZ = np.load(invZ_path)
        return invZ

    def get_batch(self, batch_size=32):
        RGB_batch = np.zeros((self.max_episode_length, batch_size, self.resolution, self.resolution, 3), dtype=np.float32)
        invZ_batch = np.zeros((self.max_episode_length, batch_size, self.resolution, self.resolution, 1), dtype=np.float32)
        mask_batch = np.zeros((self.max_episode_length, batch_size, self.resolution, self.resolution, 1), dtype=np.float32)
        sn_batch = np.zeros((self.max_episode_length, batch_size, self.resolution, self.resolution, 3), dtype=np.float32)
        vox_batch = np.zeros((batch_size, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
            dtype=np.float32)
        azim_batch = np.zeros(self.max_episode_length, batch_size, 1), dtype=np.float32)
        elev_batch = np.zeros(self.max_episode_length, batch_size, 1), dtype=np.float32)
        actions_batch = np.zeros(self.max_episode_length-1, batch_size, 1), dtype=np.float32))

        for b_idx in range(batch_size):
            higher_bound = min(self.count, self.mem_length)
            rand_idx = np.random.randint(0, higher_bound)
            data_ = self.mem_list[rand_idx]

            azim_batch[:, b_idx, :] = np.asarray(data_.state[0])
            elev_batch[:, b_idx, :] = np.asarray(data_.state[1])
            actions_batch[:, b_idx, :] = np.asarray(data_.actions)

            for l_idx in range(self.max_episode_length):
                RGB_batch[l_idx, b_idx, ...], mask_batch[l_idx, b_idx, ...] = read_png_to_uint8(
                    azim_batch[l_idx, b_idx, ...], elev_batch[l_idx, b_idx, ...], data_.model_id)

                invZ_batch[l_idx, b_idx, ...] = read_invZ(azim_batch[l_idx, b_idx, ...],
                    elev_batch[l_idx, b_idx, ...], data_.model_id)

                ## TODO: update sn_batch and vox_batch

        return RGB_batch, invZ_batch, mask_batch, sn_batch, vox_batch, azim_batch, elev_batch, actions_batch
