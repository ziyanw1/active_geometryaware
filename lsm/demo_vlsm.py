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
import scipy.io as sio
from utils import Bunch, get_session_config
import scipy.ndimage as ndimg

sys.path.append(os.path.join('../utils'))
from util import downsample
import binvox_rw

SAMPLE_DIR = os.path.join('data', 'shapenet_sample')
im_dir = os.path.join(SAMPLE_DIR, 'renders')
log_dir = os.path.join('models_lsm_v1/vlsm-release/train')
with open(os.path.join(log_dir, 'args.json'), 'r') as f:
    args = json.load(f)
    args = Bunch(args)

# Set voxel resolution
voxel_resolution = 32
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
#print net.im_batch
#sys.exit()

from shapenet import ShapeNet
# Read data
dset = ShapeNet(im_dir=im_dir, split_file=os.path.join(SAMPLE_DIR, 'splits_sample.json'), rng_seed=1)
test_mids = dset.get_smids('test')
print test_mids[0]

# Run the last three cells to run on different inputs
rand_sid, rand_mid = random.choice(test_mids) # Select model to test
rand_views = np.random.choice(dset.num_renders, size=(net.im_batch, ), replace=False) # Select views of model to test
#rand_views = range(5)
rand_sid = '03001627'
#rand_mid = '41d9bd662687cf503ca22f17e86bab24'
rand_mid = '53180e91cd6651ab76e29c9c43bc7aa'

# Load images and cameras
ims = dset.load_func['im'](rand_sid, rand_mid, rand_views)
ims = np.expand_dims(ims, 0)
R = dset.load_func['R'](rand_sid, rand_mid, rand_views)
R = np.expand_dims(R, 0)
K = dset.load_func['K'](rand_sid, rand_mid, rand_views)
K = np.expand_dims(K, 0)

# Run VLSM
print 'category: {}, model_id: {}'.format(rand_sid, rand_mid)
print 'random views: {}'.format(rand_views)
feed_dict = {net.K: K, net.Rcam: R, net.ims: ims}
pred_voxels = sess.run(net.prob_vox, feed_dict=feed_dict)
#sio.savemat('vlsm_vox.mat', {'voxels':pred_voxels[0], 'RGB_list': ims, 'R_list': R, 'K_list': K})

def read_bv(fn):
    with open(fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    data = np.float32(model.data)
    return data

def read_vox(vox_name): 
    vox_model = read_bv(vox_name) 
    vox_factor = voxel_resolution * 1.0 / 128
    #vox_model_zoom = ndimg.zoom(vox_model, vox_factor, order=0) # nearest neighbor interpolation
    vox_model_zoom = downsample(vox_model, int(1/vox_factor))
        
    vox_model_zoom = np.transpose(vox_model_zoom, (0,2,1))

    return vox_model_zoom

def calu_IoU_loss(vox_pred, vox_gt):
    
    vox_pred_bin = np.asarray(vox_pred > 0.5, dtype=np.int32)
    def calu_IoU(a, b):
        inter = a*b
        sum_inter = np.sum(inter[:])
        union = a + b
        union[union > 0.5] = 1
        sum_union = np.sum(union[:])
        return sum_inter*1.0/sum_union
    
    def calu_cross_entropy(a, b):
        a[b == 0] = 1 - a[b == 0] 
        a += 1e-5

        cross_entropy = np.log(a)
        return -np.sum(cross_entropy[:])
    
    IoU = calu_IoU(vox_pred_bin, vox_gt)
    loss = calu_cross_entropy(vox_pred, vox_gt)
    return IoU, loss

vox_name = os.path.join('../voxels', '{}/{}/model.binvox'.format(rand_sid, rand_mid))
vox_gt = read_vox(vox_name)
IoUs = []
losses = []

pred_voxels = np.squeeze(pred_voxels)
for vox_single in pred_voxels:
    iou_temp, loss_temp = calu_IoU_loss(vox_single, vox_gt)
    IoUs.append(iou_temp)
    losses.append(loss_temp)

print 'loss: {}'.format(losses)
print 'IoUs: {}'.format(IoUs)
sio.savemat('gt_vox.mat', {'voxels':vox_gt[None, ...]})
