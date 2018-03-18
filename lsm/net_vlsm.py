import os
import json
import sys
import random
sys.path.insert(0, '.')
import tensorflow as tf
import numpy as np
from models import grid_nets, im_nets, model_vlsm
from ops import conv_rnns
from mvnet import MVNet
from utils import Bunch, get_session_config

SAMPLE_DIR = os.path.join('data', 'shapenet_sample')
im_dir = os.path.join(SAMPLE_DIR, 'renders')
log_dir = os.path.join('lsm/models_lsm_v1/vlsm-release/train')
with open(os.path.join(log_dir, 'args.json'), 'r') as f:
    args = json.load(f)
    args = Bunch(args)


# Setup TF graph and initialize VLSM model
tf.reset_default_graph()

# Change the ims_per_model to run on different number of views
bs, ims_per_model = 1, 4

ckpt = 'mvnet-100000'
net = MVNet(vmin=-0.5, vmax=0.5, vox_bs=bs,
    im_bs=ims_per_model, grid_size=args.nvox,
    im_h=args.im_h, im_w=args.im_w,
    norm=args.norm, mode="TEST")

net = model_vlsm(net, im_nets[args.im_net], grid_nets[args.grid_net], conv_rnns[args.rnn])
vars_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MVNet')
sess = tf.InteractiveSession(config=get_session_config())
saver = tf.train.Saver(var_list=vars_restore)
saver.restore(sess, os.path.join(log_dir, ckpt))

#from shapenet import ShapeNet
## Read data
#dset = ShapeNet(im_dir=im_dir, split_file=os.path.join(SAMPLE_DIR, 'splits_sample.json'), rng_seed=1)
#test_mids = dset.get_smids('test')
#
## Run the last three cells to run on different inputs
#rand_sid, rand_mid = random.choice(test_mids) # Select model to test
#rand_views = np.random.choice(dset.num_renders, size=(net.im_batch, ), replace=False) # Select views of model to test
#
## Load images and cameras
#ims = dset.load_func['im'](rand_sid, rand_mid, rand_views)
#ims = np.expand_dims(ims, 0)
#R = dset.load_func['R'](rand_sid, rand_mid, rand_views)
#R = np.expand_dims(R, 0)
#K = dset.load_func['K'](rand_sid, rand_mid, rand_views)
#K = np.expand_dims(K, 0)
#
## Run VLSM
#feed_dict = {net.K: K, net.Rcam: R, net.ims: ims}
#pred_voxels = sess.run(net.prob_vox, feed_dict=feed_dict)
