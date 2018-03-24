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
import other

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
        
        self.azimuth_list_batch = tf.placeholder(dtype=tf.float32, 
            shape=[FLAGS.batch_size, FLAGS.max_episode_length, 1], name='azimuth_list_batch')
        self.elevation_list_batch = tf.placeholder(dtype=tf.float32, 
            shape=[FLAGS.batch_size, FLAGS.max_episode_length, 1], name='elevation_list_batch')
        
        self.action_list_batch = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.max_episode_length-1, 1], name='action_batch')
        ## inputs/outputs for aggregtor

        ## inputs/outputs for MVnet
        self.vox_batch = tf.placeholder(dtype=tf.float32,
            shape=[FLAGS.batch_size, FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution],
            name='vox_batch') ## [BS, V, V, V]
        self.vox_test = tf.placeholder(dtype=tf.float32,
            shape=[1, FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution],
            name='vox_batch') ## [BS, V, V, V]
        self.vox_list_batch = tf.tile(tf.expand_dims(self.vox_batch, axis=1), [1, FLAGS.max_episode_length, 1 ,1 ,1]) ##[BS, EP, V, V, V]
        self.vox_list_test = tf.tile(tf.expand_dims(self.vox_test, axis=1), [1, FLAGS.max_episode_length, 1 ,1 ,1]) ##[BS, EP, V, V, V]

        ## TEST: inputs for policy network of active camera
        self.RGB_list_test = tf.placeholder(dtype=tf.float32,
            shape=[1, FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 3], name='RGB_list_test')
        self.invZ_list_test = tf.placeholder(dtype=tf.float32,
            shape=[1, FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1], name='invZ_list_test')
        self.mask_list_test = tf.placeholder(dtype=tf.float32,
            shape=[1, FLAGS.max_episode_length, FLAGS.resolution, FLAGS.resolution, 1], name='mask_list_test')

        self.azimuth_list_test = tf.placeholder(dtype=tf.float32, 
            shape=[1, FLAGS.max_episode_length, 1], name='azimuth_list_test')
        self.elevation_list_test = tf.placeholder(dtype=tf.float32, 
            shape=[1, FLAGS.max_episode_length, 1], name='elevation_list_test')
        
        self._create_network()
        self._create_loss()

        if FLAGS.is_training:
            self._create_optimizer()
        self._create_summary()
        
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
                
                net_rgb = slim.conv2d(rgb, 16, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv1')
                net_rgb = slim.conv2d(net_rgb, 32, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv2')
                net_rgb = slim.conv2d(net_rgb, 64, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv3')
                net_rgb = slim.conv2d(net_rgb, 64, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv4')
                net_rgb = slim.conv2d(net_rgb, 128, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='rgb_conv5')
                net_rgb = slim.flatten(net_rgb, scope='rgb_flatten')

                net_vox = slim.conv3d(vox, 16, kernel_size=3, stride=2, padding='SAME', scope='vox_conv1')
                net_vox = slim.conv3d(net_vox, 32, kernel_size=3, stride=2, padding='SAME', scope='vox_conv2')
                net_vox = slim.conv3d(net_vox, 64, kernel_size=3, stride=2, padding='SAME', scope='vox_conv3')
                net_vox = slim.conv3d(net_vox, 128, kernel_size=3, stride=2, padding='SAME', scope='vox_conv4')
                net_vox = slim.conv3d(net_vox, 256, kernel_size=3, stride=2, padding='SAME', scope='vox_conv5')
                net_vox = slim.flatten(net_vox, scope='vox_flatten')
                
                net_feat = tf.concat([net_rgb, net_vox], axis=1)
                net_feat = slim.fully_connected(net_feat, 2048, scope='fc6')
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

                net_down1 = slim.conv3d(vox_feat, 16, kernel_size=4, stride=2, padding='SAME', scope='unet_conv1')
                net_down2 = slim.conv3d(net_down1, 32, kernel_size=4, stride=2, padding='SAME', scope='unet_conv2')
                net_down3 = slim.conv3d(net_down2, 64, kernel_size=4, stride=2, padding='SAME', scope='unet_conv3')
                net_down4 = slim.conv3d(net_down3, 128, kernel_size=4, stride=2, padding='SAME', scope='unet_conv4')
                net_down5 = slim.conv3d(net_down4, 256, kernel_size=4, stride=2, padding='SAME', scope='unet_conv5')

                net_up4 = slim.conv3d_transpose(net_down5, 128, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv4')
                net_up4_ = tf.concat([net_up4, net_down4], axis=-1)
                net_up3 = slim.conv3d_transpose(net_up4_, 64, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv3')
                net_up3_ = tf.concat([net_up3, net_down3], axis=-1)
                net_up2 = slim.conv3d_transpose(net_up3_, 32, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv2')
                net_up2_ = tf.concat([net_up2, net_down2], axis=-1)
                net_up1 = slim.conv3d_transpose(net_up2_, 16, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv1')
                net_up1_ = tf.concat([net_up1, net_down1], axis=-1)
                #net_out_ = slim.conv3d(net_up1_, 1, kernel_size=4, stride=2, padding='SAME', \
                #    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='unet_deconv_out')
                ## heavy load
                net_up0 = slim.conv3d_transpose(net_up1_, channels, kernel_size=4, stride=2, padding='SAME', \
                    scope='unet_deconv0')
                net_up0_ = tf.concat([net_up0, vox_feat], axis=-1)
                net_out_ = slim.conv3d(net_up0_, 1, kernel_size=3, stride=1, padding='SAME', \
                    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='unet_deconv_out')
                ## heavy load
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
                
                net_unproj = slim.conv3d(unproj_grids, 16, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv1')
                #net_unproj = slim.conv3d(net_unproj, 64, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv2')
                #net_unproj = slim.conv3d(net_unproj, 64, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv3')
        
                ## the input for convgru should be in shape of [bs, episode_len, vox_reso, vox_reso, vox_reso, ch]
                #net_unproj = uncollapse_dims(net_unproj, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
                net_unproj = uncollapse_dims(net_unproj, unproj_grids.get_shape().as_list()[0]/self.FLAGS.max_episode_length, 
                    self.FLAGS.max_episode_length)
                net_pool_grid, _ = convgru(net_unproj, filters=channels) ## should be shape of [bs, len, vox_reso x 3, ch] 

        return net_pool_grid

    def _create_policy_net(self):
        self.rgb_batch_norm = tf.subtract(self.rgb_batch, 0.5)
        self.action_prob, self.logits = self._create_dqn_two_stream(self.rgb_batch_norm, self.vox_batch,
            if_bn=self.FLAGS.if_bn, scope_name='dqn_two_stream')

    def _create_network(self):
        self.RGB_list_batch_norm = tf.subtract(self.RGB_list_batch, 0.5)
        self.RGB_list_test_norm = tf.subtract(self.RGB_list_test, 0.5)

        ## TODO: unproj depth list and merge them using aggregator
        ## collapse data from [BS, EP, H, W, CH] to [BSxEP, H, W, CH]
        ## --------------- train -------------------
        self.invZ_batch = collapse_dims(self.invZ_list_batch)
        self.mask_batch = collapse_dims(self.mask_list_batch)
        self.RGB_batch_norm = collapse_dims(self.RGB_list_batch_norm)
        self.azimuth_batch = collapse_dims(self.azimuth_list_batch)
        self.elevation_batch = collapse_dims(self.elevation_list_batch)        
        ## --------------- train -------------------
        ## --------------- test  -------------------
        self.invZ_test = collapse_dims(self.invZ_list_test)
        self.mask_test = collapse_dims(self.mask_list_test)
        self.RGB_test_norm = collapse_dims(self.RGB_list_test_norm)
        self.azimuth_test = collapse_dims(self.azimuth_list_test)
        self.elevation_test = collapse_dims(self.elevation_list_test)        
        ## --------------- test  -------------------
        with tf.device('/gpu:0'):
            ## [BSxEP, V, V, V, CH]
            self.unproj_grid_batch = self.unproj_net.unproject_batch(self.invZ_batch, self.mask_batch, self.RGB_batch_norm, self.azimuth_batch, self.elevation_batch)
            self.unproj_grid_test = self.unproj_net.unproject_batch(self.invZ_test, self.mask_test, self.RGB_test_norm, self.azimuth_test, self.elevation_test)
        
        ## TODO: collapse vox feature and do inference using unet3d
        with tf.device('/gpu:1'):
            ## --------------- train -------------------
            self.vox_feat_list = self._create_aggregator64(self.unproj_grid_batch, channels=7,
                trainable=True, if_bn=self.FLAGS.if_bn, scope_name='aggr_64') ## [BS, EP, V, V, V, CH], channels should correspond with unet_3d

            self.vox_feat = collapse_dims(self.vox_feat_list) ## [BSxEP, V, V, V, CH]
            self.vox_pred, vox_logits = self._create_unet3d(self.vox_feat, channels=7,
                trainable=True, if_bn=self.FLAGS.if_bn, scope_name='unet_3d') ## [BSxEP, V, V, V, 1], channels should correspond with aggregator
            self.vox_list_logits = uncollapse_dims(vox_logits, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
            ## --------------- train -------------------
            ## --------------- test  -------------------
            self.vox_feat_list_test = self._create_aggregator64(self.unproj_grid_test, channels=7,
                if_bn=self.FLAGS.if_bn, reuse=tf.AUTO_REUSE, scope_name='aggr_64')

            self.vox_feat_test = collapse_dims(self.vox_feat_list_test)
            self.vox_pred_test, vox_test_logits = self._create_unet3d(self.vox_feat_test, channels=7,
                if_bn=self.FLAGS.if_bn, reuse=tf.AUTO_REUSE, scope_name='unet_3d')
            self.vox_list_test_logits = uncollapse_dims(vox_test_logits, 1, self.FLAGS.max_episode_length)
            ## --------------- test  -------------------
        
        ## create active agent
        with tf.device('/gpu:0'):
            ## extract input from list [BS, EP, ...] to [BS, EP-1, ...] as we do use episode end to train
            ## --------------- train -------------------
            self.RGB_list_batch_norm_use, _ = tf.split(self.RGB_list_batch_norm, 
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            self.vox_feat_list_use, _ = tf.split(self.vox_feat_list, 
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            ## collapse input for easy inference instead of inference multiple times
            self.RGB_use_batch = collapse_dims(self.RGB_list_batch_norm_use)
            self.vox_feat_use = collapse_dims(self.vox_feat_list_use)
            self.action_prob, _ = self._create_dqn_two_stream(self.RGB_use_batch, self.vox_feat_use,
                trainable=True, if_bn=self.FLAGS.if_bn, scope_name='dqn_two_stream')
            ## --------------- train -------------------
            ## --------------- test  -------------------
            self.RGB_list_test_norm_use, _ = tf.split(self.RGB_list_test_norm,
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            self.vox_feat_list_test_use, _ = tf.split(self.vox_feat_list_test,
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            ## collapse input for easy inference instead of inference multiple times
            self.RGB_use_test = collapse_dims(self.RGB_list_test_norm_use)
            self.vox_feat_test_use = collapse_dims(self.vox_feat_list_test_use)
            self.action_prob_test, _ = self._create_dqn_two_stream(self.RGB_use_test, self.vox_feat_test_use,
                if_bn=self.FLAGS.if_bn, reuse=tf.AUTO_REUSE, scope_name='dqn_two_stream')
            ## --------------- test  -------------------

    
    def _create_loss(self):
        ## create reconstruction loss
        ## --------------- train -------------------

        ground_truth_voxels = self.vox_list_batch #rotate this
       
        if not self.FLAGS.use_coef:
            recon_loss_mat = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_voxels, 
                logits=tf.squeeze(self.vox_list_logits, axis=-1), name='recon_loss_mat')
        else:
            ## add coefficient to postive samples
            recon_loss_mat = tf.nn.weighted_cross_entropy_with_logits(targets=ground_truth_voxels, 
                logits=tf.squeeze(self.vox_list_logits, axis=-1), pos_weight=self.FLAGS.loss_coef, name='recon_loss_mat')
        self.recon_loss_list = tf.reduce_mean(recon_loss_mat, axis=[2, 3, 4], name='recon_loss_list') ## [BS, EP, V, V, V]
        self.recon_loss = tf.reduce_sum(self.recon_loss_list, axis=[0, 1], name='recon_loss')
        ## --------------- train -------------------
        ## --------------- test  -------------------
        if not self.FLAGS.use_coef:
            recon_loss_mat_test = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.vox_list_test,
                logits=tf.squeeze(self.vox_list_test_logits, axis=-1), name='recon_loss_mat_test')
        else:
            ## add coefficient to postive samples
            recon_loss_mat_test = tf.nn.weighted_cross_entropy_with_logits(targets=self.vox_list_test,
                logits=tf.squeeze(self.vox_list_test_logits, axis=-1), pos_weight=self.FLAGS.loss_coef, name='recon_loss_mat_test')
        recon_loss_mat_test = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.vox_list_test,
            logits=tf.squeeze(self.vox_list_test_logits, axis=-1), name='recon_loss_mat_test')
        self.recon_loss_list_test = tf.reduce_mean(recon_loss_mat_test, axis=[2,3,4], name='recon_loss_list_test')
        self.recon_loss_test = tf.reduce_sum(self.recon_loss_list_test, name='recon_loss_test')
        ## --------------- test  -------------------

        def process_loss_to_reward(loss_list_batch, gamma, max_episode_len, r_name=None):
            
            reward_raw_batch = loss_list_batch[:, :-1]-loss_list_batch[:, 1:]
            reward_batch_list = tf.get_variable(name='reward_batch_list_{}'.format(r_name), shape=reward_raw_batch.get_shape(),
                dtype=tf.float32, initializer=tf.zeros_initializer)

            batch_size = loss_list_batch.get_shape().as_list()[0]
            
            ## decayed sum of future possible rewards
            for i in range(max_episode_len):
                for j in range(i, max_episode_len):
                    update_r = reward_raw_batch[:, j] * (gamma**(j-i))
                    update_r = update_r + reward_batch_list[:, i] 
                    update_r = tf.expand_dims(update_r, axis=1)
                    ## update reward batch list
                    reward_batch_list = tf.concat(axis=1, values=[reward_batch_list[:, :i], update_r,
                        reward_batch_list[:,i+1:]])

            return 10*reward_batch_list, 10*reward_raw_batch

        self.reward_batch_list, self.reward_raw_batch = process_loss_to_reward(self.recon_loss_list, self.FLAGS.gamma,
            self.FLAGS.max_episode_length-1)
        self.reward_test_list, self.reward_raw_test = process_loss_to_reward(self.recon_loss_list_test, self.FLAGS.gamma,
            self.FLAGS.max_episode_length-1, r_name='test')
            
        ## create reinforce loss
        self.action_batch = collapse_dims(self.action_list_batch)
        self.indexes = tf.range(0, tf.shape(self.action_prob)[0]) * tf.shape(self.action_prob)[1] + self.action_batch
        self.responsible_action = tf.gather(tf.reshape(self.action_prob, [-1]), self.indexes)
        ## reward_batch node should not back propagate
        self.reward_batch = tf.stop_gradient(collapse_dims(self.reward_batch_list), name='reward_batch')
        self.loss_reinforce = -tf.reduce_mean(tf.log(self.responsible_action)*self.reward_batch, name='reinforce_loss')

    def _create_optimizer(self):
       
        aggr_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='aggr')
        unet_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet')
        dqn_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dqn')

        if self.FLAGS.if_constantLr:
            self.learning_rate = self.FLAGS.learning_rate
            #self._log_string(tf_util.toGreen('===== Using constant lr!'))
        else:  
            self.learning_rate = get_learning_rate(self.counter, self.FLAGS)

        if self.FLAGS.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.FLAGS.momentum)
        elif self.FLAGS.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        #self.opt_recon = self.optimizer.minimize(self.recon_loss, var_list=aggr_var+unet_var, global_step=self.counter)  
        #self.opt_reinforce = self.optimizer.minimize(self.loss_reinforce, var_list=aggr_var+dqn_var,
        #    global_step=self.counter)
        self.opt_recon = self.optimizer.minimize(self.recon_loss, var_list=aggr_var+unet_var)  
        self.opt_reinforce = self.optimizer.minimize(self.loss_reinforce, var_list=aggr_var+dqn_var)

    def _create_summary(self):
        if self.FLAGS.is_training:
            self.summary_learning_rate = tf.summary.scalar('train/learning_rate', self.learning_rate)
            self.summary_loss_recon_train = tf.summary.scalar('train/loss_recon',
                self.recon_loss/(self.FLAGS.max_episode_length*self.FLAGS.batch_size))
            self.summary_loss_reinforce_train = tf.summary.scalar('train/loss_reinforce', self.loss_reinforce)
            self.summary_reward_batch_train = tf.summary.scalar('train/reward_batch', tf.reduce_sum(self.reward_batch))

            self.merge_train_list = [self.summary_learning_rate, self.summary_loss_recon_train,
                self.summary_loss_reinforce_train, self.summary_reward_batch_train]
            self.merged_train = tf.summary.merge(self.merge_train_list)

    def get_placeholders(self, include_vox, include_action, train_mode):
        
        placeholders = lambda: None
        if train_mode:
            placeholders.rgb = self.RGB_list_batch
            placeholders.invz = self.invZ_list_batch
            placeholders.mask = self.mask_list_batch
            placeholders.azimuth = self.azimuth_list_batch
            placeholders.elevation = self.elevation_list_batch

            if include_action:
                placeholders.action = self.action_list_batch
            if include_vox:
                placeholders.vox = self.vox_batch

        else:
            placeholders.rgb = self.RGB_list_test
            placeholders.invz = self.invZ_list_test
            placeholders.mask = self.mask_list_test
            placeholders.azimuth = self.azimuth_list_test
            placeholders.elevation = self.elevation_list_test

            if include_action:
                placeholders.action = self.action_list_test
            if include_vox:
                placeholders.vox = self.vox_test

        return placeholders

    def construct_feed_dict(self, mvnet_inputs, include_vox, include_action, train_mode = True):

        placeholders = self.get_placeholders(include_vox, include_action, train_mode = train_mode)

        feed_dict = {self.is_training: train_mode}

        keys = ['rgb', 'invz', 'mask', 'azimuth', 'elevation']
        if include_vox:
            assert mvnet_inputs.vox is not None
            keys.append('vox')
        if include_action:
            assert mvnet_inputs.action is not None
            keys.append('action')
            
        for key in keys:

            ph_input = getattr(mvnet_inputs, key)
            if not train_mode:
                ph_input = ph_input[None, ...]

            feed_dict[getattr(placeholders, key)] = ph_input

        return feed_dict

    def select_action(self, mvnet_input, idx, is_training = False):
        
        feed_dict = self.construct_feed_dict(
            mvnet_input, include_vox = False, include_action = False, train_mode = is_training
        )
    
        #if np.random.uniform(low=0.0, high=1.0) > epsilon:
        #    action_prob = self.sess.run([self.action_prob], feed_dict=feed_dict)
        #else:
        #    return np.random.randint(low=0, high=FLAGS.action_num)
        stuff = self.sess.run([self.action_prob_test], feed_dict=feed_dict)
        action_prob = stuff[0][idx]
        if is_training:
            a_response = np.random.choice(action_prob, p=action_prob)

            a_idx = np.argmax(action_prob == a_response)
        else:
            a_idx = np.argmax(action_prob)
        return a_idx

    def predict_vox_list(self, mvnet_input, is_training = False):

        feed_dict = self.construct_feed_dict(
            mvnet_input, include_vox = True, include_action = False, train_mode = is_training
        )
        
        #if np.random.uniform(low=0.0, high=1.0) > epsilon:
        #    action_prob = self.sess.run([self.action_prob], feed_dict=feed_dict)
        #else:
        #    return np.random.randint(low=0, high=FLAGS.action_num)
        vox_test_list, recon_loss_list, rewards_test = self.sess.run([
            self.vox_pred_test, self.recon_loss_list_test,
            self.reward_raw_test], feed_dict=feed_dict)
        
        return vox_test_list, recon_loss_list, rewards_test

    def run_step(self, mvnet_input, mode, is_training = True):
        '''mode is one of 'burnin', 'train' '''
        
        feed_dict = self.construct_feed_dict(
            mvnet_input, include_vox = True, include_action = True, train_mode = is_training
        )

        if mode == 'burnin':
            ops_to_run = [
                self.unproj_grid_batch,
                self.opt_recon,
                self.recon_loss,
                self.recon_loss_list,
                self.action_prob,
                self.reward_batch_list,
                self.reward_raw_batch,
                self.loss_reinforce,
            ]
        elif mode == 'train':
            ops_to_run = [
                self.unproj_grid_batch,
                self.recon_loss,
                self.loss_reinforce,
                self.recon_loss_list,
                self.action_prob,
                self.reward_batch_list,
                self.reward_raw_batch,
                self.opt_recon,
                self.opt_reinforce,
                self.merged_train
            ]     
        else:
            assert 'bad mode'
        
        out_stuff = self.sess.run(ops_to_run, feed_dict=feed_dict)
        return out_stuff
        

class MVInput(object):
    def __init__(self, rgb, invz, mask, azimuth, elevation, vox = None, action = None):
        self.rgb = rgb
        self.invz = invz
        self.mask = mask
        self.azimuth = azimuth
        self.elevation = elevation
        self.vox = vox
        self.action = action

