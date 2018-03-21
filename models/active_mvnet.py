import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import util
from utils import tf_util

from env_data.replay_memory import ReplayMemory
from env_data.shapenet_env import ShapeNetEnv, trajectData  
from lsm.ops import convgru, convlstm, collapse_dims, uncollapse_dims 
from util_unproj import unproject_tools 

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1*x + f2 * abs(x)

class ActiveMVnet(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        #self.senv = ShapeNetEnv(FLAGS)
        #self.replay_mem = ReplayMemory(FLAGS)
        self.unproj_net = unproject_tools(FLAGS)

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.activation_fn = lrelu
        self.counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        ## inputs/outputs for policy network of active camera
        self.RGB_list_batch = tf.placeholder(dtype=tf.float32, 
            shape=[FLAGS.batch_size, FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3], name='RGB_list_batch')
        self.invZ_list_batch = tf.placeholder(dtype=tf.float32, 
            shape=[FLAGS.batch_size, FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1], name='invZ_list_batch')
        self.mask_list_batch = tf.placeholder(dtype=tf.float32, 
            shape=[FLAGS.batch_size, FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1], name='mask_list_batch')
        self.action_list_batch = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.max_episode_length-1], name='action_batch')
        self.reward_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.max_episode_length-1], name='reward_batch')
        ## inputs/outputs for aggregtor

        ## inputs/outputs for MVnet
        self.vox_batch = tf.placeholder(dtype=tf.float32,
            shape=[FLAGS.batch_size, FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution],
            name='vox_batch') ## [BS, V, V, V]
        self.vox_list_batch = tf.tile(tf.expand_dims(self.vox_batch, axis=1), [1, FLAGS.max_episode_length, 1 ,1 ,1]) ##[BS, EP, V, V, V]

        self._create_network()
        self._create_loss()
        #self._create_policy_net()
        #self._create_loss()

        if FLAGS.is_training:
            self._create_optimizer()
        #self._create_summary()
        
        # Add ops to save and restore all variable 
        self.saver = tf.train.Saver()
        
        # create a sess
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.LOG_DIR, 'train'), self.sess.graph)

    def _create_dqn_two_stream(self, rgb, vox, trainable=True, if_bn=False, reuse=False, scope_name='dqn_two_stream'):
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
                
                net_rgb = slim.conv2d(rgb, 64, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv1')
                net_rgb = slim.conv2d(net_rgb, 128, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv2')
                net_rgb = slim.conv2d(net_rgb, 256, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv3')
                net_rgb = slim.conv2d(net_rgb, 256, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv4')
                net_rgb = slim.conv2d(net_rgb, 256, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv5')
                net_rgb = slim.flatten(net_rgb, scope='rgb_flatten')

                net_vox = slim.conv3d(vox, 64, kernel_size=3, stride=1, padding='SAME', scope='vox_conv1')
                net_vox = slim.conv3d(net_vox, 128, kernel_size=3, stride=2, padding='SAME', scope='vox_conv2')
                net_vox = slim.conv3d(net_vox, 256, kernel_size=3, stride=1, padding='SAME', scope='vox_conv3')
                net_vox = slim.conv3d(net_vox, 256, kernel_size=3, stride=2, padding='SAME', scope='vox_conv4')
                net_vox = slim.conv3d(net_vox, 512, kernel_size=3, stride=2, padding='SAME', scope='vox_conv5')
                net_vox = slim.flatten(net_vox, scope='vox_flatten')
                
                net_feat = tf.concat([net_rgb, net_vox], axis=1)
                net_feat = slim.fully_connected(net_feat, 4096, scope='fc6')
                net_feat = slim.fully_connected(net_feat, 4096, scope='fc7')
                logits = slim.fully_connected(net_feat, self.FLAGS.action_num, activation_fn=None, scope='fc8')

                return tf.nn.softmax(logits), logits
    
    def _create_unet3d(self, vox_feat, channels, trainable=True, if_bn=False, reuse=False, scope_name='unet_3d'):

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

                net_down1 = slim.conv3d(vox_feat, 64, kernel_size=4, stride=2, padding='SAME', scope='unet_conv1')
                net_down2 = slim.conv3d(net_down1, 128, kernel_size=4, stride=2, padding='SAME', scope='unet_conv2')
                net_down3 = slim.conv3d(net_down2, 256, kernel_size=4, stride=2, padding='SAME', scope='unet_conv3')
                net_down4 = slim.conv3d(net_down3, 512, kernel_size=4, stride=2, padding='SAME', scope='unet_conv4')
                net_down5 = slim.conv3d(net_down4, 512, kernel_size=4, stride=2, padding='SAME', scope='unet_conv5')

                net_up4 = slim.conv3d_transpose(net_down5, 512, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv4')
                net_up4_ = tf.concat([net_up4, net_down4], axis=-1)
                net_up3 = slim.conv3d_transpose(net_up4_, 256, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv3')
                net_up3_ = tf.concat([net_up3, net_down3], axis=-1)
                net_up2 = slim.conv3d_transpose(net_up3_, 128, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv2')
                net_up2_ = tf.concat([net_up2, net_down2], axis=-1)
                net_up1 = slim.conv3d_transpose(net_up2_, 64, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv1')
                net_up1_ = tf.concat([net_up1, net_down1], axis=-1)
                net_up0 = slim.conv3d_transpose(net_up1_, channels, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv0')
                net_up0_ = tf.concat([net_up0, vox_feat], axis=-1)
                net_out_ = slim.conv3d_transpose(net_up0_, 1, kernel_size=3, stride=1, padding='SAME', \
                    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='unet_deconv_out')
                #net_up2_ = tf.add(net_up2, net_down2)
                #net_up1 = slim.conv3d_transpose(net_up2_, 64, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                #    scope='unet_deconv1')
                #net_up1_ = tf.concat([net_up1, net_down1], axis=-1)
                #net_out_ = slim.conv3d_transpose(net_up1_, out_channel, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                #    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='unet_out')
            
        return tf.nn.sigmoid(net_out_), net_out_

    def _create_aggregator64(self, unproj_grids, channels, trainable=True, if_bn=False, reuse=False, scope_name='aggr_64'):
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
            
            ## create fuser
            with slim.arg_scope([slim.fully_connected],
                    activation_fn=self.activation_fn,
                    trainable=trainable,
                    normalizer_fn=batch_normalizer_gen,
                    normalizer_params=batch_norm_params_gen,
                    weights_regularizer=weights_regularizer):
                
                net_unproj = slim.conv3d(unproj_grids, 32, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv1')
                #net_unproj = slim.conv3d(net_unproj, 64, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv2')
                #net_unproj = slim.conv3d(net_unproj, 64, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv3')
        
                ## the input for convgru should be in shape of [bs, episode_len, vox_reso, vox_reso, vox_reso, ch]
                net_unproj = uncollapse_dims(net_unproj, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
                net_pool_grid, _ = convgru(net_unproj, filters=channels) ## should be shape of [bs, len, vox_reso x 3, ch] 

        return net_pool_grid

    def _create_policy_net(self):
        self.rgb_batch_norm = tf.subtract(self.rgb_batch, 0.5)
        self.action_prob, self.logits = self._create_dqn_two_stream(self.rgb_batch_norm, self.vox_batch,
            if_bn=self.FLAGS.if_bn, scope_name='dqn_two_stream')

    def _create_network(self):
        self.RGB_list_batch_norm = tf.subtract(self.RGB_list_batch, 0.5)

        ## TODO: unproj depth list and merge them using aggregator
        ## collapse data from [BS, EP, H, W, CH] to [BSxEP, H, W, CH]
        self.invZ_batch = collapse_dims(self.invZ_list_batch)
        self.mask_batch = collapse_dims(self.mask_list_batch)
        self.RGB_batch_norm = collapse_dims(self.RGB_list_batch_norm)
        with tf.device('/gpu:0'):
            ## [BSxEP, V, V, V, CH]
            self.unproj_grid_batch = self.unproj_net.unproject_batch(self.invZ_batch, self.mask_batch, self.RGB_batch_norm)
            
        
        ## TODO: collapse vox feature and do inference using unet3d
        with tf.device('/gpu:1'):
            self.vox_feat_list = self._create_aggregator64(self.unproj_grid_batch, channels=7, trainable=self.is_training,
                if_bn=self.FLAGS.if_bn, scope_name='aggr_64') ## [BS, EP, V, V, V, CH], channels should correspond with unet_3d

            self.vox_feat = collapse_dims(self.vox_feat_list) ## [BSxEP, V, V, V, CH]
            self.vox_pred, vox_logits = self._create_unet3d(self.vox_feat, channels=7, trainable=self.is_training,
                if_bn=self.FLAGS.if_bn, scope_name='unet_3d') ## [BSxEP, V, V, V, 1], channels should correspond with aggregator
            self.vox_list_logits = uncollapse_dims(vox_logits, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
        
        ## create active agent
        with tf.device('/gpu:0'):
            ## extract input from list [BS, EP, ...] to [BS, EP-1, ...] as we do use episode end to train
            self.RGB_list_batch_norm_use, _ = tf.split(self.RGB_list_batch_norm, 
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            self.vox_feat_list_use, _ = tf.split(self.vox_feat_list, 
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            ## collapse input for easy inference instead of inference multiple times
            self.RGB_use_batch = collapse_dims(self.RGB_list_batch_norm_use)
            self.vox_feat_use = collapse_dims(self.vox_feat_list_use)
            self.action_prob, _ = self._create_dqn_two_stream(self.RGB_use_batch, self.vox_feat_use,
                trainable=self.is_training, if_bn=self.FLAGS.if_bn, scope_name='dqn_two_stream')

    
    def _create_loss(self):
        ## create reconstruction loss
        recon_loss_mat = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.vox_list_batch,
            logits=tf.squeeze(self.vox_list_logits, axis=-1), name='recon_loss_mat')
        self.recon_loss_list = tf.reduce_mean(recon_loss_mat, axis=[0, 2, 3, 4], name='recon_loss_list') ## [BS, EP, V, V, V]
        self.recon_loss = tf.reduce_sum(self.recon_loss_list, name='recon_loss')
        ## create reinforce loss
        self.action_batch = collapse(self.action_list_batch)
        self.indexes = tf.range(0, tf.shape(self.action_prob)[0]) * tf.shape(self.action_prob)[1] + self.action_batch
        self.responsible_action = tf.gather(tf.reshape(self.action_prob, [-1]), self.indexes)
        #self.loss = -tf.reduce_mean(tf.log(self.responsible_action)*self.reward_batch, name='reinforce_loss')

    def _create_optimizer(self):
       
        aggr_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='aggr')
        unet_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet')
        if self.FLAGS.if_constantLr:
            self.learning_rate = self.FLAGS.learning_rate
            #self._log_string(tf_util.toGreen('===== Using constant lr!'))
        else:  
            self.learning_rate = get_learning_rate(self.counter, self.FLAGS)

        if self.FLAGS.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.FLAGS.momentum)
        elif self.FLAGS.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.opt_recon = self.optimizer.minimize(self.recon_loss, var_list=aggr_var+unet_var, global_step=self.counter)  

    #def _create_summary(self):
    #    if self.FLAGS.is_training:
    #        self.summary_learning_rate = tf.summary.scalar('train/learning_rate', self.learning_rate)
    #        self.summary_loss_train = tf.summary.scalar('train/loss', self.loss)

    #        self.merge_train_list = [self.summary_learning_rate, self.summary_loss_train]
    #        self.merged_train = tf.summary.merge(self.merge_train_list)
