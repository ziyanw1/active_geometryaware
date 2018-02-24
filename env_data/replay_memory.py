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
        mask = np.tile(np.expand_dims(mask, 2), (1, 1,3))
        new_img = new_img * mask + np.ones_like(new_img, dtype=np.float32) * (1.0 - mask)
        return (new_img*255.).astype(np.uint8), mask
