import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
import numpy as np
import math
import sys
import os
import other

import tensorflow.contrib.layers as ly
from utils import util
from utils import tf_util
from rgb2d_lmdb_loader import data_loader 

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1*x + f2 * abs(x)


class AE_rgb2d(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        #self.data_loader = data_loader(self.FLAGS)

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        self.counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.test_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.activation_fn = lrelu

        self.global_i = tf.Variable(0, name='global_i', trainable=False)
        self.set_i_to_pl = tf.placeholder(tf.int32,shape=[], name='set_i_to_pl')
        self.assign_i_op = tf.assign(self.global_i, self.set_i_to_pl)
        
        self.rgb_batch = tf.placeholder(tf.float32, shape=(None,FLAGS.resolution,FLAGS.resolution,3), name='input_rgb')
        self.invZ_batch = tf.placeholder(tf.float32, shape=(None,FLAGS.resolution,FLAGS.resolution,1), name='gt_invZ')
        self.mask_batch = tf.placeholder(tf.float32, shape=(None,FLAGS.resolution,FLAGS.resolution,1), name='gt_mask')
        self.sn_batch = tf.placeholder(tf.float32, shape=(None,FLAGS.resolution,FLAGS.resolution,3), name='gt_sn')

        
        self._create_network()
        self._create_loss()
        
        #self._create_optimizer()
        #self._create_summary()
        
        # Add ops to save and restore all variable 
        self.saver = tf.train.Saver()
        self.restorer = tf.train.Saver()
        
        # create a sess
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # # Start input enqueue threads.
        self.coord = tf.train.Coordinator()
        print "===== main-->tf.train.start_queue_runners"
        #self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        # create summary
        self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.LOG_DIR, 'train'), self.sess.graph)

    #def _visualize(self):
    #    print '='*10
    #    print self.voxel_batch
    #    print self.pred_voxels
    #    print self.rgb_batch
    #    print self.depth_batch
    #    print self.mask_batch
    #    print '='*10        
    #    
    #    self.vis = {
    #        'voxel': self.voxel_batch,
    #        'pred_voxel': self.pred_voxels,
    #        'rgb': self.rgb_batch,
    #        'depth': self.depth_batch,
    #        'mask': self.mask_batch,
    #        'gt_voxel2depth': self.gt_voxel2depth,
    #        'gt_voxel2mask': self.gt_voxel2mask, 
    #    }

    #def _voxel_pred(self):

    #    self.voxel_batch = tf.expand_dims(self.data_loader.voxel_batch, axis = 4)
    #    self.depth_batch = 2.0/self.invZ_batch #scale by x2 to make the depths ~4

    #    BG_DEPTH = 4.0
    #    bg_depth = tf.ones_like(self.depth_batch)*BG_DEPTH
    #    self.depth_batch = tf.where(self.depth_batch > 5.0, bg_depth, self.depth_batch)

    #    #using gt inputs for now...
    #    in_depth = self.depth_batch
    #    in_mask = self.mask_batch

    #    in_depth -= 4.0 #subtract the baseline

    #    #mask **must** come before depth
    #    pred_inputs = tf.concat([in_mask, in_depth], axis = 3)

    #    #setting up some constants
    #    other.constants.S = self.FLAGS.voxel_resolution

    #    fov = 30.0
    #    focal_length = 1.0/math.tan(fov*math.pi/180/2)
    #    other.constants.focal_length = focal_length
    #    other.constants.BS = self.FLAGS.batch_size

    #    other.constants.DEBUG_UNPROJECT = False
    #    other.constants.USE_LOCAL_BIAS = False
    #    other.constants.USE_OUTLINE = True
    #    
    #    pred_inputs = other.nets.unproject(pred_inputs)
    #    other.constants.mode = 'train'
    #    other.constants.rpvx_unsup = False
    #    other.constants.force_batchnorm_trainmode = False
    #    other.constants.force_batchnorm_testmode = False
    #    other.constants.NET3DARCH = 'marr'
    #    
    #    pred_outputs = self._voxel_net(pred_inputs)

    #    self.pred_voxels = pred_outputs

    #    self.angles = self.data_loader.angles_batch
    #    self.theta = -self.angles[:,0] #remember that we're flipping theta!
    #    self.phi = self.angles[:,1]

    #    print self.theta
    #    print self.phi
    #    world2cam_rot_mat = other.voxel.get_transform_matrix_tf(self.theta, self.phi)
    #    print world2cam_rot_mat

    #    gt_voxel = other.voxel.transformer_preprocess(self.voxel_batch)
    #    gt_voxel = other.voxel.rotate_voxel(gt_voxel, world2cam_rot_mat)
    #    self.gt_voxel = gt_voxel

    #    print gt_voxel

    #    proj_and_post = lambda x: other.voxel.transformer_postprocess(other.voxel.project_voxel(x))
    #    projected_gt = proj_and_post(gt_voxel)
    #    print projected_gt
    #    
    #    def flatten(voxels):
    #        H = self.FLAGS.resolution
    #        W = self.FLAGS.resolution            
    #        
    #        pred_depth = other.voxel.voxel2depth_aligned(voxels)
    #        pred_mask = other.voxel.voxel2mask_aligned(voxels)
    #    
    #        #replace bg with grey
    #        hard_mask = tf.cast(pred_mask > 0.5, tf.float32)
    #        pred_depth *= hard_mask

    #        BG_DEPTH = 3.0
    #        pred_depth += BG_DEPTH * (1.0 - hard_mask)

    #        pred_depth = tf.image.resize_images(pred_depth, (H, W))
    #        pred_mask = tf.image.resize_images(pred_mask, (H, W))
    #        return pred_depth, pred_mask
    #    
    #    gt_depth, gt_mask = flatten(projected_gt)

    #    self.gt_voxel2depth = gt_depth
    #    self.gt_voxel2mask = gt_mask

    #    #take losses
    #    other.constants.eps = 1E-6

    #    other.constants.DEBUG_LOSSES = False
    #    if other.constants.DEBUG_LOSSES:
    #        self.pred_voxels = other.tfpy.summarize_tensor(self.pred_voxels, 'pred')
    #    
    #    self.voxel_loss = other.losses.binary_ce_loss(self.pred_voxels, self.gt_voxel)
    #    
    #def _voxel_net(self, pred_inputs):
    #    if other.constants.DEBUG_UNPROJECT:
    #        return pred_inputs
    #    else:
    #        with tf.variable_scope('voxel_net'):
    #            out = other.nets.voxel_net_3d(pred_inputs)
    #            self.voxel_net_3d_vars = other.tfutil.current_scope_and_vars()[1]
    #        return out
        
    def _create_unet(self, rgb, out_channel=1, trainable=True, if_bn=False, reuse=False, scope_name='unet_2d'):

        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            if if_bn:
                batch_normalizer_gen = slim.batch_norm
                batch_norm_params_gen = {'is_training': self.is_training, 'decay': self.FLAGS.bn_decay}
            else:
                #self._print_arch('=== NOT Using BN for GENERATOR!')
                batch_normalizer_gen = None
                batch_norm_params_gen = None

            if self.FLAGS.if_l2Reg:
                weights_regularizer = slim.l2_regularizer(1e-5)
            else:
                weights_regularizer = None


            with slim.arg_scope([slim.fully_connected],
                    activation_fn=self.activation_fn,
                    trainable=trainable,
                    normalizer_fn=batch_normalizer_gen,
                    normalizer_params=batch_norm_params_gen,
                    weights_regularizer=weights_regularizer):

                net_down1 = slim.conv2d(rgb, 64, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='unet_conv1')
                net_down2 = slim.conv2d(net_down1, 128, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='unet_conv2')
                net_down3 = slim.conv2d(net_down2, 256, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='unet_conv3')
                net_down4 = slim.conv2d(net_down3, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='unet_conv4')
                net_down5 = slim.conv2d(net_down4, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='unet_conv5')
                net_down6 = slim.conv2d(net_down5, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='unet_conv6')
                net_bottleneck = slim.conv2d(net_down6, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='unet_conv7')

                net_up6 = slim.conv2d_transpose(net_bottleneck, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv6')
                net_up6_ = tf.concat([net_up6, net_down6], axis=-1)
                net_up5 = slim.conv2d_transpose(net_up6_, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv5')
                net_up5_ = tf.concat([net_up5, net_down5], axis=-1)

                net_up4 = slim.conv2d_transpose(net_up5_, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv4')
                net_up4_ = tf.concat([net_up4, net_down4], axis=-1)
                net_up3 = slim.conv2d_transpose(net_up4_, 256, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv3')
                net_up3_ = tf.concat([net_up3, net_down3], axis=-1)
                net_up2 = slim.conv2d_transpose(net_up3_, 128, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv2')
                net_up2_ = tf.add(net_up2, net_down2)
                net_up1 = slim.conv2d_transpose(net_up2_, 64, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv1')
                net_up1_ = tf.concat([net_up1, net_down1], axis=-1)
                net_out_ = slim.conv2d_transpose(net_up1_, out_channel, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='unet_out')

            
        return net_out_, net_bottleneck
    
    def _create_encoder(self, rgb, trainable=True, if_bn=False, reuse=False, scope_name='unet_encoder'):

        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            if if_bn:
                batch_normalizer_gen = slim.batch_norm
                batch_norm_params_gen = {'is_training': self.is_training, 'decay': self.FLAGS.bn_decay}
            else:
                #self._print_arch('=== NOT Using BN for GENERATOR!')
                batch_normalizer_gen = None
                batch_norm_params_gen = None

            if self.FLAGS.if_l2Reg:
                weights_regularizer = slim.l2_regularizer(1e-5)
            else:
                weights_regularizer = None


            with slim.arg_scope([slim.fully_connected],
                    activation_fn=self.activation_fn,
                    trainable=trainable,
                    normalizer_fn=batch_normalizer_gen,
                    normalizer_params=batch_norm_params_gen,
                    weights_regularizer=weights_regularizer):

                net_down1 = slim.conv2d(rgb, 64, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='ae_conv1')
                net_down2 = slim.conv2d(net_down1, 128, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='ae_conv2')
                net_down3 = slim.conv2d(net_down2, 256, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='ae_conv3')
                net_down4 = slim.conv2d(net_down3, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='ae_conv4')
                net_down5 = slim.conv2d(net_down4, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='ae_conv5')
                net_down6 = slim.conv2d(net_down5, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='ae_conv6')
                net_bottleneck = slim.conv2d(net_down6, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', scope='ae_conv7')

        return net_bottleneck
    
    def _create_decoder(self, z_rgb, out_channel=1, trainable=True, if_bn=False, reuse=False, scope_name='unet_decoder'):

        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            if if_bn:
                batch_normalizer_gen = slim.batch_norm
                batch_norm_params_gen = {'is_training': self.is_training, 'decay': self.FLAGS.bn_decay}
            else:
                #self._print_arch('=== NOT Using BN for GENERATOR!')
                batch_normalizer_gen = None
                batch_norm_params_gen = None

            if self.FLAGS.if_l2Reg:
                weights_regularizer = slim.l2_regularizer(1e-5)
            else:
                weights_regularizer = None


            with slim.arg_scope([slim.fully_connected],
                    activation_fn=self.activation_fn,
                    trainable=trainable,
                    normalizer_fn=batch_normalizer_gen,
                    normalizer_params=batch_norm_params_gen,
                    weights_regularizer=weights_regularizer):

                net_up6 = slim.conv2d_transpose(z_rgb, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv6')
                net_up5 = slim.conv2d_transpose(net_up6, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv5')
                net_up4 = slim.conv2d_transpose(net_up5, 512, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv4')
                net_up3 = slim.conv2d_transpose(net_up4, 256, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv3')
                net_up2 = slim.conv2d_transpose(net_up3, 128, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv2')
                net_up1 = slim.conv2d_transpose(net_up2, 64, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    scope='unet_deconv1')
                net_out_ = slim.conv2d_transpose(net_up1, out_channel, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='unet_out')

        return net_out_

    def _create_network(self):
        with tf.device('/gpu:0'):

            self.rgb_batch_norm = tf.subtract(tf.div(self.rgb_batch, 255.), 0.5)

            #self.rgb_batch = tf.placeholder(tf.float32, shape=(None,128,128,3), name='input_rgb')
            #self.invZ_batch = tf.placeholder(tf.float32, shape=(None,128,128,1), name='gt_invZ')
            
            #self.invZ_pred, self.z_rgb = self._create_unet(self.rgb_batch_norm, trainable=True, if_bn=True, scope_name='unet_rgb2depth') 
            if self.FLAGS.network_name == 'ae':
                self.z_rgb = self._create_encoder(self.rgb_batch_norm, trainable=True, if_bn=True, scope_name='unet_encoder') 
                self.preds = self._create_decoder(self.z_rgb, out_channel=5, trainable=True, if_bn=True, scope_name='unet_decoder')
            elif self.FLAGS.network_name == 'unet':
                self.preds, self.z_rgb = self._create_unet(self.rgb_batch_norm, out_channel=5, trainable=True,
                    if_bn=True, scope_name='unet_rgb2depth') 
            else:
                raise NotImplementedError

            self.invZ_pred, self.mask_pred, self.sn_pred = tf.split(self.preds, [1,1,3], axis=3, name='split')


    def _create_loss(self):
        self.depth_recon_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.invZ_pred-self.invZ_batch),
            [1,2,3])), 0, name='depth_recon_loss')
        
        self.sn_recon_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.sn_pred-self.sn_batch),
            [1,2,3])), 0, name='sn_recon_loss')

        self.mask_cls_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask_batch, \
            logits=self.mask_pred), [1,2,3]), 0, name='mask_cls_loss')

        self.sketch_loss = self.depth_recon_loss + self.sn_recon_loss + self.mask_cls_loss

    def _create_optimizer(self):
        
        unet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet')
        if self.FLAGS.if_constantLr:
            self.learning_rate = self.FLAGS.learning_rate
            #self._log_string(tf_util.toGreen('===== Using constant lr!'))
        else:  
            self.learning_rate = get_learning_rate(self.counter, self.FLAGS)

        if self.FLAGS.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.FLAGS.momentum)
        elif self.FLAGS.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
        self.opt_step = self.optimizer.minimize(
            self.sketch_loss, var_list=unet_vars, global_step=self.counter
        )
        

    def _create_summary(self):

        self.summary_learning_rate = tf.summary.scalar('train/learning_rate', self.learning_rate)
        self.summary_loss_train = tf.summary.scalar('train/loss', self.sketch_loss)
        self.summary_loss_test = tf.summary.scalar('test/loss', self.sketch_loss)

        self.summary_loss_depth_recon_train = tf.summary.scalar('train/loss_depth_recon', self.depth_recon_loss)
        self.summary_loss_depth_recon_test = tf.summary.scalar('test/loss_depth_recon', self.depth_recon_loss)
        self.summary_loss_sn_recon_train = tf.summary.scalar('train/loss_sn_recon', self.sn_recon_loss)
        self.summary_loss_sn_recon_test = tf.summary.scalar('test/loss_sn_recon', self.sn_recon_loss)
        self.summary_loss_mask_cls_train = tf.summary.scalar('train/loss_mask_cls', self.mask_cls_loss)
        self.summary_loss_mask_cls_test = tf.summary.scalar('test/loss_mask_cls', self.mask_cls_loss)

        self.summary_train = [self.summary_loss_train, self.summary_loss_depth_recon_train,
            self.summary_loss_sn_recon_train, self.summary_loss_mask_cls_train, self.summary_learning_rate]
        self.summary_test = [self.summary_loss_test, self.summary_loss_depth_recon_test,
            self.summary_loss_sn_recon_test, self.summary_loss_mask_cls_test]

        self.merge_train = tf.summary.merge(self.summary_train)
        self.merge_test = tf.summary.merge(self.summary_test)
