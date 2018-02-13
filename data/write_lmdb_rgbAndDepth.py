#!/usr/bin/env python2

# scp jerrypiglet@128.237.133.169:Bitsync/3dv2017_PBA/data/write_lmdb_imageAndShape_direct2.py . && CUDA_VISIBLE_DEVICES=3 python write_lmdb_imageAndShape_direct2.py --ae_file '/newfoundland/rz1/log/finalAE_1e-5_bnNObn_car24576__bb10/'

import numpy as np
import math
import os, sys
import os.path
import scipy.io as sio
import scipy.ndimage as ndimg
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

import binvox_rw

#global FLAGS
#flags = tf.flags
#flags.DEFINE_string('ae_file', '', '')
#flags.DEFINE_boolean('if_en_bn', True, 'If use batch normalization for the mesh decoder')
#flags.DEFINE_boolean('if_gen_bn', False, 'If use batch normalization for the mesh generator')
#flags.DEFINE_float('bn_decay', 0.9, 'Decay rate for batch normalization [default: 0.9]')
#FLAGS = flags.FLAGS

sample_num = 24576
#resolution=32
resolution = 128
VIEWS = 200
vox_factor = 0.25

#BASE_OUT_DIR = '/home/rz1/Documents/Research/3dv2017_PBA_out/'
BASE_OUT_DIR = './data_cache'

# pcd_path = '/home/rz1/Documents/Research/3dv2017_PBA_out/PCDs/'

#LMDB_DIR='./lmdb'
LMDB_DIR='./lmdb128'

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

voxel_dir = '../voxels' 

def read_bv(fn):
    with open(fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    data = np.float32(model.data)
    return data

class lmdb_writer(DataFlow):
    def __init__(self, model_ids):
        self.model_ids = model_ids

    def get_data(self):
        for model_id in self.model_ids:

            try:
                assert os.path.exists(os.path.join(render_out_path,model_id))
            except:
                print 'warning -- skipping model: %s' % model_id
                continue
                
            
            #ply_name = pcd_path + '%s/%s_%d.ply'%(category_name, model_id, sample_num)
            # default resolution: 128x128x128
            vox_name = os.path.join(voxel_dir, '{}/{}/model.binvox'.format(category_name, model_id)) 
            mat_name = render_out_path + '/%s_tw.mat'%model_id
            vox_model = read_bv(vox_name)
            vox_model_zoom = ndimg.zoom(vox_model, vox_factor, order=0) # nearest neighbor interpolation
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
                    surfaceNormal_single = get_surface_from_depth(invZ_single)
                    #print mask_single.shape
                    #print mask_single.dtype
                    if invZ_single.ndim == 2:
                        invZ_single = invZ_single[:, :, None]
                    if mask_single.ndim == 2:
                        mask_single = mask_single[:, :, None]
                    if mask_single.shape[2] == 3:
                        mask_single = mask_single[:, :, 0]
                        mask_single = mask_single[:, :, None]

                    yield [rgb_single, invZ_single, mask_single, surfaceNormal_single, \
                        np.asarray([a, e, 0.], dtype=np.float32), vox_model_zoom.astype(np.uint8)]

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
    depth = np.reciprocal(invZ)
    depth[depth == np.inf] = 4
    dz_dy = np.gradient(depth, axis=0)
    dz_dx = np.gradient(depth, axis=1)
    
    d = np.concatenate((-dz_dx[:, :, None], -dz_dy[:, :, None], np.ones_like(depth[:, :, None])), axis=-1)
    l = np.linalg.norm(d, axis=-1)
    n = np.divide(d, l[:, :, None]) 

    return n

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

if __name__ == "__main__":
    
    # splits_list = [['train', 'test', 'val'], ['train', 'val'], ['test']]
    splits_list = [['train', 'val'], ['test']]
    # splits_list = [['test']]
    for category_name in categories:
        render_out_path = os.path.join(BASE_OUT_DIR, 'blender_renderings/%s/res%d_chair_debug_nonorm'%(category_name, \
            resolution))
        render_out_path = os.path.join(BASE_OUT_DIR, 'blender_renderings/%s/res%d_chair_all'%(category_name, \
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
            write_path = LMDB_DIR
            # write_path = '/data_tmp/lmdbqqqq'
            lmdb_write = write_path + "/random_randomLamp0822_%s_%d_%s_imageAndShape_single.lmdb"%(cat_name[category_name], sample_num, lmdb_name_append)
            lmdb_write = os.path.join(write_path, 'rgb2depth_single_0209.lmdb')
            # depth,mask,surfnorm,campose,vox32
            lmdb_write = os.path.join(write_path, 'rgb2depth_single_{}_0212.lmdb'.format(lmdb_name_append)) 

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
