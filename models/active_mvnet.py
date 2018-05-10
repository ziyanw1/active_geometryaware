import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import util
from utils import tf_util

from env_data.shapenet_env import ShapeNetEnv, trajectData  
from lsm.ops import convgru, convlstm, collapse_dims, uncollapse_dims 
from util_unproj import Unproject_tools 
import other
from tensorflow import summary as summ
from tensorflow.python import debug as tf_debug

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
        self.unproj_net = Unproject_tools(FLAGS)

        self.activation_fn = lrelu
        self.counter = tf.Variable(0, trainable=False, dtype=tf.int32)

        self._create_placeholders()
        self._create_ground_truth_voxels()
        self._create_network()
        self._create_loss()
        #if FLAGS.is_training:
        self._create_optimizer()
        self._create_summary()
        self._create_collections()
        
        # Add ops to save and restore all variable 
        self.saver = tf.train.Saver()
        if FLAGS.initial_dqn:
            aggr_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='aggr')
            unet_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet')
            dqn_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dqn')
            self.pretrain_saver = tf.train.Saver(unet_var+aggr_var, max_to_keep=None)
        else:
            self.pretrain_saver = tf.train.Saver(max_to_keep=None)
        
        # create a sess
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        ### for debug
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, dump_root='./tfdbg')
        ### for debug
        self.sess.run(tf.global_variables_initializer())

        self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.LOG_DIR, 'train'), self.sess.graph)

    def _create_placeholders(self):
        
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        self.train_provider = ShapeProvider(self.FLAGS)
        self.test_provider = ShapeProvider(self.FLAGS, batch_size = 1)

        self.train_provider.make_tf_ph()
        self.test_provider.make_tf_ph()
        
        self.RGB_list_batch = self.train_provider.rgb_ph
        self.invZ_list_batch = self.train_provider.invz_ph
        self.mask_list_batch = self.train_provider.mask_ph
        self.azimuth_list_batch = self.train_provider.azimuth_ph
        self.elevation_list_batch = self.train_provider.elevation_ph
        self.action_list_batch = self.train_provider.action_ph
        self.penalty_list_batch = self.train_provider.penalty_ph
        self.vox_batch = self.train_provider.vox_ph
        self.seg1_batch = self.train_provider.seg1_ph
        self.seg2_batch = self.train_provider.seg2_ph        
        
        self.RGB_list_test = self.test_provider.rgb_ph
        self.invZ_list_test = self.test_provider.invz_ph
        self.mask_list_test = self.test_provider.mask_ph
        self.azimuth_list_test = self.test_provider.azimuth_ph
        self.elevation_list_test = self.test_provider.elevation_ph
        self.action_list_test = self.test_provider.action_ph
        self.penalty_list_test = self.test_provider.penalty_ph
        self.vox_test = self.test_provider.vox_ph
        self.seg1_test = self.test_provider.seg1_ph
        self.seg2_test = self.test_provider.seg2_ph

    def _create_ground_truth_voxels(self):
        az0_train = self.azimuth_list_batch[:,0,0]
        el0_train = self.elevation_list_batch[:,0,0]
        az0_test = self.azimuth_list_test[:,0,0]
        el0_test = self.elevation_list_test[:,0,0]

        def rotate_voxels(vox, az0, el0):
            vox = tf.expand_dims(vox, axis = 4)
            #negative sign is important -- although i'm not sure how it works
            R = other.voxel.get_transform_matrix_tf(-az0, el0)
            #print vox.get_shape().as_list()
            return tf.clip_by_value(other.voxel.rotate_voxel(other.voxel.transformer_preprocess(vox), R), 0.0, 1.0)

        def tile_voxels(x):
            return tf.tile(
                tf.expand_dims(x, axis = 1),
                [1, self.FLAGS.max_episode_length, 1 ,1 ,1, 1]
            )

        self.rotated_vox_batch = rotate_voxels(self.vox_batch, az0_train, el0_train)
        self.rotated_vox_test = rotate_voxels(self.vox_test, az0_test, el0_test)
        
        self.rotated_vox_list_batch = tile_voxels(self.rotated_vox_batch)
        self.rotated_vox_list_test = tile_voxels(self.rotated_vox_test)
        self.vox_list_batch = tile_voxels(self.vox_batch[..., None])
        self.vox_list_test = tile_voxels(self.vox_test[..., None])

        if self.FLAGS.debug_mode:
            summ.histogram('ground_truth_voxels', self.rotated_vox_batch)
        
    def _create_dqn_two_stream(self, rgb, vox, trainable=True, if_bn=False, reuse=False,
                               scope_name='dqn_two_stream'):
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            
            if if_bn:
                batch_normalizer_gen = slim.batch_norm
                batch_norm_params_gen = {'is_training': self.is_training, 
                                         'decay': self.FLAGS.bn_decay,
                                         'epsilon': 1e-5,
                                         'scale': True,
                                         'updates_collections': None,
                                         'trainable': trainable}
                #batch_norm_params_gen = {'is_training': True, 'decay': self.FLAGS.bn_decay}
                #batch_normalizer_gen = None
                #batch_norm_params_gen = None
            else:
                #self._print_arch('=== NOT Using BN for GENERATOR!')
                batch_normalizer_gen = None
                batch_norm_params_gen = None

            if self.FLAGS.if_dqn_l2Reg:
                weights_regularizer = slim.l2_regularizer(1e-5)
            else:
                weights_regularizer = None
            
            with slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv3d],
                    activation_fn=self.activation_fn,
                    trainable=trainable,
                    normalizer_fn=batch_normalizer_gen,
                    normalizer_params=batch_norm_params_gen,
                    weights_regularizer=weights_regularizer):
                
                net_rgb = slim.conv2d(rgb, 32, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='rgb_conv1')
                net_rgb = slim.conv2d(net_rgb, 32, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='rgb_conv2')
                net_rgb = slim.conv2d(net_rgb, 64, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='rgb_conv3')
                net_rgb = slim.conv2d(net_rgb, 64, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='rgb_conv4')
                net_rgb = slim.conv2d(net_rgb, 128, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='rgb_conv5')
                net_rgb = slim.flatten(net_rgb, scope='rgb_flatten')

                vox = tf.stop_gradient(tf.identity(vox), name='dqn_vox')
                #vox = slim.batch_norm(vox, decay=self.FLAGS.bn_decay, scale=True, epsilon=1e-5,
                #    updates_collections=None, is_training=self.is_training, trainable=trainable)
                net_vox = slim.conv3d(vox, 16, kernel_size=3, stride=2, padding='VALID', scope='vox_conv1')
                net_vox = slim.conv3d(net_vox, 32, kernel_size=3, stride=2, padding='VALID', scope='vox_conv2')
                net_vox = slim.conv3d(net_vox, 32, kernel_size=3, stride=2, padding='VALID', scope='vox_conv3')
                net_vox = slim.conv3d(net_vox, 64, kernel_size=3, stride=2, padding='VALID', scope='vox_conv4')
                net_vox = slim.conv3d(net_vox, 128, kernel_size=3, stride=2, padding='VALID', scope='vox_conv5')
                net_vox = slim.flatten(net_vox, scope='vox_flatten')
                
                net_feat = tf.concat([net_rgb, net_vox], axis=1)
                net_feat = slim.fully_connected(net_feat, 4096, scope='fc6')
                net_feat = slim.fully_connected(net_feat, 4096, scope='fc7')
                logits = slim.fully_connected(net_feat, self.FLAGS.action_num, activation_fn=None, normalizer_fn=None, scope='fc8')
                values = slim.fully_connected(net_feat, 1, activation_fn=None, normalizer_fn=None, scope='fc8_value')

                return tf.nn.softmax(logits), logits, values
    
    def _create_dqn(self, vox, trainable=True, if_bn=False, reuse=False, scope_name='dqn'):
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            
            if if_bn:
                batch_normalizer_gen = slim.batch_norm
                batch_norm_params_gen = {'is_training': self.is_training, 
                                         'decay': self.FLAGS.bn_decay,
                                         'epsilon': 1e-5,
                                         'scale': True,
                                         'updates_collections': None,
                                         'trainable': trainable}
                #batch_norm_params_gen = {'is_training': True, 'decay': self.FLAGS.bn_decay}
                #batch_normalizer_gen = None
                #batch_norm_params_gen = None
            else:
                #self._print_arch('=== NOT Using BN for GENERATOR!')
                batch_normalizer_gen = None
                batch_norm_params_gen = None

            if self.FLAGS.if_dqn_l2Reg:
                weights_regularizer = slim.l2_regularizer(1e-5)
            else:
                weights_regularizer = None
            
            with slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv3d],
                    activation_fn=self.activation_fn,
                    trainable=trainable,
                    normalizer_fn=batch_normalizer_gen,
                    normalizer_params=batch_norm_params_gen,
                    weights_regularizer=weights_regularizer):
                
                vox = tf.stop_gradient(tf.identity(vox), name='dqn_vox')
                net_vox = slim.conv3d(vox, 16, kernel_size=3, stride=2, padding='VALID', scope='vox_conv1')
                net_vox = slim.conv3d(net_vox, 32, kernel_size=3, stride=2, padding='VALID', scope='vox_conv2')
                net_vox = slim.conv3d(net_vox, 32, kernel_size=3, stride=2, padding='VALID', scope='vox_conv3')
                net_vox = slim.conv3d(net_vox, 64, kernel_size=3, stride=2, padding='VALID', scope='vox_conv4')
                net_vox = slim.conv3d(net_vox, 128, kernel_size=3, stride=2, padding='VALID', scope='vox_conv5')
                net_vox = slim.flatten(net_vox, scope='vox_flatten')
                
                net_feat = slim.fully_connected(net_vox, 4096, scope='fc6')
                net_feat = slim.fully_connected(net_feat, 4096, scope='fc7')
                logits = slim.fully_connected(net_feat, self.FLAGS.action_num, activation_fn=None, normalizer_fn=None, scope='fc8')
                values = slim.fully_connected(net_feat, 1, activation_fn=None, normalizer_fn=None, scope='fc8_value')

                return tf.nn.softmax(logits), logits, values
    
    def _create_unet3d(self, vox_feat, mask, channels, trainable=True, if_bn=False, reuse=False, scope_name='unet_3d'):

        if self.FLAGS.unet_name == 'U_SAME':
            return other.nets.unet_same(
                vox_feat, channels, self.FLAGS, trainable = trainable, if_bn = if_bn, reuse = reuse,
                is_training = self.is_training, activation_fn = self.activation_fn, scope_name = scope_name
            )
        elif self.FLAGS.unet_name == 'U_VALID':
            #can't run summaries for test
            #debug = int(vox_feat.get_shape().as_list()[0]) > self.FLAGS.max_episode_length
            debug = False
            with tf.variable_scope(scope_name, reuse = reuse):
                return other.nets.voxel_net_3d_v2(vox_feat, bn = if_bn, bn_trainmode=self.is_training, 
                    freeze_decoder=not trainable, return_logits = True, debug = debug)
        elif self.FLAGS.unet_name == 'U_VALID_SPARSE':
            return other.nets.unet_valid_sparese(
                vox_feat, mask, channels, self.FLAGS, trainable = trainable, if_bn = if_bn, reuse = reuse,
                is_training = self.is_training, activation_fn = self.activation_fn, scope_name = scope_name
            )
        elif self.FLAGS.unet_name == 'OUTLINE':
            return vox_feat, tf.zeros_like(vox_feat)
        else:
            raise Exception, 'not a valid unet name'
    
    def _create_aggregator64(self, unproj_grids, channels, trainable=True, if_bn=False, reuse=False,
                             scope_name='aggr_64'):

        if self.FLAGS.agg_name == 'GRU':
            return other.nets.gru_aggregator(
                unproj_grids, channels, self.FLAGS, trainable = trainable, if_bn = if_bn, reuse = reuse,
                is_training = self.is_training, activation_fn = self.activation_fn, scope_name = scope_name
            )
        elif self.FLAGS.agg_name == 'POOL':
            return other.nets.pooling_aggregator(
                unproj_grids, channels, self.FLAGS, trainable = trainable, reuse = reuse,
                is_training = self.is_training, scope_name = scope_name
            )
        elif self.FLAGS.agg_name == 'MAXPOOL':
            return other.nets.max_pooling_aggregator(
                unproj_grids, channels, self.FLAGS, trainable = trainable, reuse = reuse,
                is_training = self.is_training, scope_name = scope_name
            )
        elif self.FLAGS.unet_name == 'OUTLINE':
            #bs = int(unproj_grids.get_shape()[0]) / self.FLAGS.max_episode_length
            #unproj_grids = uncollapse_dims(unproj_grids, bs, self.FLAGS.max_episode_length)
            rvals = [tf.reduce_max(unproj_grids[:,:i+1,:,:,:,-1:], axis = 1)
                     for i in range(self.FLAGS.max_episode_length)]
            return tf.stack(rvals, axis = 1)
        else:
            raise Exception, 'not a valid agg name'

    def _create_policy_net(self):
        self.rgb_batch_norm = tf.subtract(self.rgb_batch, 0.5)
        self.action_prob, self.logits, _ = self._create_dqn_two_stream(self.rgb_batch_norm, self.vox_batch,
            if_bn=self.FLAGS.if_bn, scope_name='dqn_two_stream')

    def _create_network(self):
        self.RGB_list_batch_norm = tf.subtract(self.RGB_list_batch, 0.5)
        self.RGB_list_test_norm = tf.subtract(self.RGB_list_test, 0.5)

        ## TODO: unproj depth list and merge them using aggregator
        ## collapse data from [BS, EP, H, W, CH] to [BSxEP, H, W, CH]
        ## --------------- train -------------------
        # self.invZ_batch = collapse_dims(self.invZ_list_batch)
        # self.mask_batch = collapse_dims(self.mask_list_batch)
        # self.RGB_batch_norm = collapse_dims(self.RGB_list_batch_norm)
        # self.azimuth_batch = collapse_dims(self.azimuth_list_batch)
        # self.elevation_batch = collapse_dims(self.elevation_list_batch)        
        # ## --------------- train -------------------
        # ## --------------- test  -------------------
        # self.invZ_test = collapse_dims(self.invZ_list_test)
        # self.mask_test = collapse_dims(self.mask_list_test)
        # self.RGB_test_norm = collapse_dims(self.RGB_list_test_norm)
        # self.azimuth_test = collapse_dims(self.azimuth_list_test)
        # self.elevation_test = collapse_dims(self.elevation_list_test)        
        ## --------------- test  -------------------
        with tf.device('/gpu:0'):
            
            ## [BSxEP, V, V, V, CH]
            if self.FLAGS.occu_only:
                self.unproj_grid_batch = self.unproj_net.unproject(
                    self.invZ_list_batch,
                    self.mask_list_batch,
                    tf.zeros_like(self.RGB_list_batch_norm),
                    self.azimuth_list_batch,
                    self.elevation_list_batch
                )

                _, self.unproj_grid_batch = tf.split(self.unproj_grid_batch, [6, 1], axis=-1)
                self.unproj_grid_mask = self.unproj_grid_batch

                self.unproj_grid_test = self.unproj_net.unproject(
                    self.invZ_list_test,
                    self.mask_list_test,
                    tf.zeros_like(self.RGB_list_test_norm),
                    self.azimuth_list_test,
                    self.elevation_list_test
                )
                _, self.unproj_grid_test = tf.split(self.unproj_grid_test, [6, 1], axis=-1)
                self.unproj_grid_mask_test = self.unproj_grid_test
            else:
                self.unproj_grid_batch = self.unproj_net.unproject(
                    self.invZ_list_batch,
                    self.mask_list_batch,
                    self.RGB_list_batch_norm,
                    self.azimuth_list_batch,
                    self.elevation_list_batch
                )
                _, self.unproj_grid_mask = tf.split(self.unproj_grid_batch, [6, 1], axis=-1)
                self.unproj_grid_mask = tf.identity(self.unproj_grid_mask)
                
                self.unproj_grid_test = self.unproj_net.unproject(
                    self.invZ_list_test,
                    self.mask_list_test,
                    self.RGB_list_test_norm,
                    self.azimuth_list_test,
                    self.elevation_list_test
                )
                _, self.unproj_grid_mask_test = tf.split(self.unproj_grid_test, [6, 1], axis=-1)
                self.unproj_grid_mask_test = tf.identity(self.unproj_grid_mask_test)

        if self.FLAGS.debug_mode:
            summ.histogram('unprojections', self.unproj_grid_batch)

        ## TODO: collapse vox feature and do inference using unet3d
        with tf.device('/gpu:1'):
            ## --------------- train -------------------
            
            ## [BS, EP, V, V, V, CH], channels should correspond with unet_3d
            self.vox_feat_list = self._create_aggregator64(
                self.unproj_grid_batch,
                channels=self.FLAGS.agg_channels,
                trainable=True,
                if_bn=self.FLAGS.if_bn,
                scope_name='aggr_64'
            )

            
            vox_feat_unstack = tf.unstack(self.vox_feat_list, axis=1)
            mask_grid_unstack = tf.unstack(self.unproj_grid_mask, axis=1)
            vox_pred_first, vox_logits_first = self._create_unet3d(
                vox_feat_unstack[0],
                mask_grid_unstack[0], 
                channels=self.FLAGS.agg_channels,
                trainable=True,
                if_bn=self.FLAGS.if_bn,
                scope_name='unet_3d'
            )

            unet_3d_reuse = lambda (x, y): self._create_unet3d(
                x,
                y,
                channels=self.FLAGS.agg_channels,
                trainable=True,
                if_bn=self.FLAGS.if_bn,
                reuse=True,
                scope_name='unet_3d'
            )
            unet_3d_reuse_test = lambda (x, y): self._create_unet3d(
                x,
                y,
                channels=self.FLAGS.agg_channels,
                trainable=False,
                if_bn=self.FLAGS.if_bn,
                reuse=True,
                scope_name='unet_3d'
            )
            vox_pred_follow, vox_logits_follow = tf.map_fn(unet_3d_reuse, (tf.stack(vox_feat_unstack[1:]), 
                tf.stack(mask_grid_unstack[1:])), dtype=(tf.float32, tf.float32))
            self.vox_list_pred = tf.stack([vox_pred_first]+tf.unstack(vox_pred_follow), axis=1)
            self.vox_list_logits = tf.stack([vox_logits_first]+tf.unstack(vox_logits_follow), axis=1)

            #self.vox_feat = collapse_dims(self.vox_feat_list) ## [BSxEP, V, V, V, CH]
            #self.vox_pred, vox_logits = self._create_unet3d(
            #    self.vox_feat,
            #    channels=self.FLAGS.agg_channels,
            #    trainable=True,
            #    if_bn=self.FLAGS.if_bn,
            #    scope_name='unet_3d'
            #)

            #self.vox_list_logits = uncollapse_dims(vox_logits, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
            #self.vox_list_pred = uncollapse_dims(self.vox_pred, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
            
            ## --------------- train -------------------
            ## --------------- test  -------------------
            self.vox_feat_list_test = self._create_aggregator64(
                self.unproj_grid_test,
                trainable=False,
                channels=self.FLAGS.agg_channels,
                if_bn=self.FLAGS.if_bn,
                reuse=True,
                scope_name='aggr_64'
            )

            vox_feat_test_unstack = tf.unstack(self.vox_feat_list_test, axis=1)
            mask_grid_test_unstack = tf.unstack(self.unproj_grid_mask_test, axis=1)

            vox_pred_test_all, vox_logits_test_all = tf.map_fn(unet_3d_reuse_test, (tf.stack(vox_feat_test_unstack), 
                tf.stack(mask_grid_test_unstack)), dtype=(tf.float32, tf.float32))

            self.vox_pred_test_ = tf.stack(tf.unstack(vox_pred_test_all), axis=1)
            self.vox_pred_test = tf.squeeze(self.vox_pred_test_)
            self.vox_list_test_logits = tf.stack(tf.unstack(vox_logits_test_all), axis=1)
            
            #self.vox_feat_test = collapse_dims(self.vox_feat_list_test)
            #self.vox_pred_test, vox_test_logits = self._create_unet3d(
            #    self.vox_feat_test,
            #    channels=self.FLAGS.agg_channels,
            #    if_bn=self.FLAGS.if_bn,
            #    reuse=tf.AUTO_REUSE,
            #    scope_name='unet_3d'
            #)
            #
            #self.vox_list_test_logits = uncollapse_dims(vox_test_logits, 1, self.FLAGS.max_episode_length)
            #self.vox_list_test_pred = uncollapse_dims(self.vox_pred_test, 1, self.FLAGS.max_episode_length)
            ## --------------- test  -------------------
            
        if self.FLAGS.debug_mode:
            summ.histogram('aggregated', self.vox_feat_list)
            summ.histogram('unet_out', self.vox_pred)
            
        ## create active agent
        with tf.device('/gpu:0'):
            ## extract input from list [BS, EP, ...] to [BS, EP-1, ...] as we do not use episode end to train
            ## --------------- train -------------------
            ### TODO:BATCH NORM ON TIME SEQ
            self.RGB_list_batch_norm_use, _ = tf.split(self.RGB_list_batch_norm, 
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            self.vox_feat_list_use, _ = tf.split(self.vox_feat_list, 
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            #RGB_list_norm = tf.unstack(self.RGB_list_batch_norm, axis=1)
            #vox_feat_list_ = tf.unstack(self.vox_feat_list, axis=1)
            ## collapse input for easy inference instead of inference multiple times
            self.RGB_use_batch = collapse_dims(self.RGB_list_batch_norm)
            self.vox_feat_use = collapse_dims(self.vox_feat_list)
            
            if self.FLAGS.dqn_use_rgb:
                self.action_prob, _, self.state_values = self._create_dqn_two_stream(self.RGB_use_batch, self.vox_feat_use,
                    trainable=True, if_bn=self.FLAGS.if_dqn_bn, scope_name='dqn_two_stream')
            else:
                self.action_prob, _, self.state_values = self._create_dqn(self.vox_feat_use, trainable=True,
                    if_bn=self.FLAGS.if_dqn_bn, scope_name='dqn_vox_only')
            self.action_prob = uncollapse_dims(self.action_prob, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
            self.action_prob = tf.stack(tf.unstack(self.action_prob, axis=1)[:-1], axis=1)
            self.action_prob = collapse_dims(self.action_prob)
            self.state_values = uncollapse_dims(self.state_values, self.FLAGS.batch_size, self.FLAGS.max_episode_length)
            self.value_batch = tf.stack(tf.unstack(self.state_values, axis=1)[:-1], axis=1)
            self.value_last_batch = tf.unstack(self.state_values, axis=1)[-1]
            self.value_batch = collapse_dims(self.value_batch)
            self.value_next_batch = tf.stack(tf.unstack(self.state_values, axis=1)[1:], axis=1)
            #self.value_next_batch = tf.identity(self.value_next_batch))
            self.value_next_batch = tf.stop_gradient(collapse_dims(self.value_next_batch))
            ### TODO:BATCH NORM ON TIME SEQ

            ### TODO:BATCH NORM ON EACH TIME STEP
            #RGB_list_norm = tf.unstack(self.RGB_list_batch_norm, axis=1)
            #vox_feat_list_ = tf.unstack(self.vox_feat_list, axis=1)
            #action_prob_first, _ = self._create_dqn_two_stream(RGB_list_norm[0], vox_feat_list_[0],
            #    trainable=True, if_bn=self.FLAGS.if_bn, scope_name='dqn_two_stream')

            #policy_net_reuse = lambda (x, y): self._create_dqn_two_stream(x, y, trainable=True, if_bn=self.FLAGS.if_bn,
            #    reuse=tf.AUTO_REUSE, scope_name='dqn_two_stream')

            #action_prob_follow, _ = tf.map_fn(policy_net_reuse, (tf.stack(RGB_list_norm[1:-1]),
            #    tf.stack(vox_feat_list_[1:-1])), dtype=(tf.float32, tf.float32)) 

            #self.action_prob = tf.stack([action_prob_first]+tf.unstack(action_prob_follow), axis=1)
            #self.action_prob = collapse_dims(self.action_prob)
            ### TODO:BATCH NORM ON EACH TIME STEP

            ## --------------- train -------------------
            ## --------------- test  -------------------
            ## TODO:BATCH NORM ON TIME SEQ
            self.RGB_list_test_norm_use, _ = tf.split(self.RGB_list_test_norm,
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            self.vox_feat_list_test_use, _ = tf.split(self.vox_feat_list_test,
                [self.FLAGS.max_episode_length-1, 1], axis=1)
            ## collapse input for easy inference instead of inference multiple times
            self.RGB_use_test = collapse_dims(self.RGB_list_test_norm)
            self.vox_feat_test_use = collapse_dims(self.vox_feat_list_test)
            if self.FLAGS.dqn_use_rgb:
                self.action_prob_test, _, self.state_values_test = self._create_dqn_two_stream(self.RGB_use_test, self.vox_feat_test_use,
                    trainable=False, if_bn=self.FLAGS.if_dqn_bn, reuse=True, scope_name='dqn_two_stream')
            else:
                self.action_prob_test, _, self.state_values_test = self._create_dqn(self.vox_feat_test_use,
                    trainable=False, if_bn=self.FLAGS.if_dqn_bn, reuse=True, scope_name='dqn_vox_only')
            self.action_prob_test = uncollapse_dims(self.action_prob_test, 1, self.FLAGS.max_episode_length)
            self.action_prob_test = tf.stack(tf.unstack(self.action_prob_test, axis=1)[:-1], axis=1)
            self.action_prob_test = collapse_dims(self.action_prob_test)
            ## TODO:BATCH NORM ON TIME SEQ
            
            ### TODO:BATCH NORM ON EACH TIME STEP
            #RGB_list_test_norm_ = tf.unstack(self.RGB_list_test_norm, axis=1)
            #vox_feat_list_test_ = tf.unstack(self.vox_feat_list_test, axis=1)
            #self.action_prob_test, _ = tf.map_fn(policy_net_reuse, (tf.stack(RGB_list_test_norm_[0:-1]),
            #    tf.stack(vox_feat_list_test_[0:-1])), dtype=(tf.float32, tf.float32))
            ### TODO:BATCH NORM ON EACH TIME STEP

            ## --------------- test  -------------------
        def rotate_voxels(vox, az0, el0):
            vox = tf.expand_dims(vox, axis = 4)
            #negative sign is important -- although i'm not sure how it works
            R = other.voxel.get_transform_matrix_tf(-az0, el0, invert_rot=True)
            #print vox.get_shape().as_list()
            return tf.clip_by_value(other.voxel.transformer_preprocess(other.voxel.rotate_voxel(vox, R)), 0.0, 1.0)
        
        az0_test = self.azimuth_list_test[:,0,0]
        el0_test = self.elevation_list_test[:,0,0]
        rotate_func = lambda x: rotate_voxels(x, az0_test, el0_test) 
        #print self.vox_pred_test.get_shape().as_list()
        self.vox_pred_test_rot = tf.map_fn(rotate_func, tf.stack(tf.unstack(tf.expand_dims(self.vox_pred_test, axis=0),
            axis=1))) 
    
    def _create_loss(self):
        ## create reconstruction loss
        ## --------------- train -------------------

        if not self.FLAGS.use_coef:
            recon_loss_mat = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.rotated_vox_list_batch, 
                #labels=self.vox_list_batch, 
                logits=self.vox_list_logits,
                name='recon_loss_mat',
            )
        else:
            recon_loss_mat = tf.nn.weighted_cross_entropy_with_logits(
                targets=self.rotated_vox_list_batch, 
                #targets=self.vox_list_batch, 
                logits=self.vox_list_logits,
                pos_weight=self.FLAGS.loss_coef,
                name='recon_loss_mat',
            )
            

        self.recon_loss_list = tf.reduce_mean(
            recon_loss_mat,
            axis=[2, 3, 4, 5],
            name='recon_loss_list'
        ) ## [BS, EP, V, V, V, 1]

        ## use last view for reconstruction
        #self.recon_loss = tf.reduce_sum(self.recon_loss_list[:, -1, ...], axis=0, name='recon_loss')
        self.recon_loss = tf.reduce_sum(self.recon_loss_list, axis=[0, 1], name='recon_loss')
        self.recon_loss_last = tf.reduce_sum(self.recon_loss_list[:, -1], axis=0, name='recon_loss_last')
        self.recon_loss_first = tf.reduce_sum(self.recon_loss_list[:, 0], axis=0, name='recon_loss_first')

        ## TODO: compute IoU
        def compute_IoU(vox_list_pred, vox_list_gt, thres=0.5, iou_name=None):
            vox_collapse_pred = collapse_dims(vox_list_pred)
            vox_collapse_gt = collapse_dims(vox_list_gt)

            def compute_IoU_single(vox_pred, vox_gt, thres):
                pred_ = tf.greater(vox_pred, thres*tf.ones_like(vox_pred, dtype=tf.float32))
                gt_ = tf.greater(vox_gt, thres*tf.ones_like(vox_gt, dtype=tf.float32))

                inter = tf.cast(tf.logical_and(pred_, gt_), dtype=tf.float32)
                union = tf.cast(tf.logical_or(pred_, gt_), dtype=tf.float32)

                return tf.div(tf.reduce_sum(inter, axis=[0,1,2]), tf.reduce_sum(union, axis=[0,1,2])+1)

            iou = lambda (x,y): compute_IoU_single(x, y, thres=thres)
            
            iou_collapse = tf.map_fn(iou, (vox_collapse_pred, vox_collapse_gt), dtype=(tf.float32))

            return iou_collapse
        
        IoU_collapse = compute_IoU(self.vox_list_pred, self.rotated_vox_list_batch, thres=self.FLAGS.iou_thres)
        self.IoU_list_batch = uncollapse_dims(IoU_collapse, self.FLAGS.batch_size, self.FLAGS.max_episode_length)

        ## --------------- train -------------------
        ## --------------- test  -------------------

        if not self.FLAGS.use_coef:
            recon_loss_mat_test = tf.nn.sigmoid_cross_entropy_with_logits(
                #labels=self.rotated_vox_list_test, 
                labels=self.vox_list_test, 
                logits=self.vox_list_test_logits,
                name='recon_loss_mat',
            )
        else:
            recon_loss_mat_test = tf.nn.weighted_cross_entropy_with_logits(
                #targets=self.rotated_vox_list_test, 
                targets=self.vox_list_test, 
                logits=self.vox_list_test_logits,
                pos_weight=self.FLAGS.loss_coef,
                name='recon_loss_mat',
            )
        
        self.recon_loss_list_test = tf.reduce_mean(
            recon_loss_mat_test,
            axis=[2,3,4,5],
            name='recon_loss_list_test'
        )
        
        self.recon_loss_test = tf.reduce_sum(self.recon_loss_list_test, name='recon_loss_test')
        IoU_collapse_test = compute_IoU(self.vox_pred_test[None, ..., None], self.rotated_vox_list_test, thres=self.FLAGS.iou_thres,
            iou_name='test')
        self.IoU_list_test = uncollapse_dims(IoU_collapse_test, 1, self.FLAGS.max_episode_length)
        ## --------------- test  -------------------


        def process_loss_to_reward(loss_list_batch, penalty_list_batch, gamma, max_episode_len, r_name='',
            reward_weight=10, penalty_weight=0.0005):
            
            #reward_raw_batch = loss_list_batch[:, :-1]-loss_list_batch[:, 1:] ## loss should be gradually decreasing
            reward_raw_batch = loss_list_batch[:, 1:] ## loss should be gradually decreasing
            penalty_use_batch = tf.squeeze(penalty_list_batch[:, 1:], axis=-1)
            reward_batch_list = tf.get_variable(name='reward_batch_list_{}'.format(r_name), shape=reward_raw_batch.get_shape(),
                dtype=tf.float32, initializer=tf.zeros_initializer)

            batch_size = loss_list_batch.get_shape().as_list()[0]
            
            ## decayed sum of future possible rewards
            for i in range(max_episode_len):
                for j in range(i, max_episode_len):
                    #update_r = reward_raw_batch[:, j]/tf.abs(loss_list_batch[:, j])*(gamma**(j-i)) - penalty_weight*penalty_use_batch[:, j]
                    update_r = reward_raw_batch[:, j]*(gamma**(j-i)) - penalty_weight*penalty_use_batch[:, j]
                    update_r = update_r + reward_batch_list[:, i] 
                    update_r = tf.expand_dims(update_r, axis=1)
                    ## update reward batch list
                    reward_batch_list = tf.concat(axis=1, values=[reward_batch_list[:, :i], update_r,
                        reward_batch_list[:,i+1:]])

            return reward_batch_list, reward_raw_batch
        
        def process_iou_to_reward(loss_list_batch, penalty_list_batch, gamma, max_episode_len, r_name=None,
            reward_weight=10, penalty_weight=0.0005):
            
            reward_raw_batch = loss_list_batch[:, 1:]-loss_list_batch[:, :-1] ## IoU should be gradually increasing
            penalty_use_batch = tf.squeeze(penalty_list_batch[:, 1:], axis=-1)
            reward_batch_list = tf.get_variable(name='reward_batch_list_{}'.format(r_name), shape=reward_raw_batch.get_shape(),
                dtype=tf.float32, initializer=tf.zeros_initializer)

            batch_size = loss_list_batch.get_shape().as_list()[0]
            
            ## decayed sum of future possible rewards
            for i in range(max_episode_len):
                for j in range(i, max_episode_len):
                    update_r = reward_raw_batch[:, j]*(gamma**(j-i)) - penalty_weight*penalty_use_batch[:, j]
                    update_r = update_r + reward_batch_list[:, i] 
                    update_r = tf.expand_dims(update_r, axis=1)
                    ## update reward batch list
                    reward_batch_list = tf.concat(axis=1, values=[reward_batch_list[:, :i], update_r,
                        reward_batch_list[:,i+1:]])

            return reward_batch_list, reward_raw_batch

        if self.FLAGS.reward_type == 'IG':
            self.reward_batch_list, self.reward_raw_batch = process_loss_to_reward(
                self.recon_loss_list,
                self.penalty_list_batch,
                self.FLAGS.gamma,
                self.FLAGS.max_episode_length-1,
                r_name=None,
                reward_weight=self.FLAGS.reward_weight,
                penalty_weight=self.FLAGS.penalty_weight
            )
            
            self.reward_test_list, self.reward_raw_test = process_loss_to_reward(
                self.recon_loss_list_test,
                self.penalty_list_test,
                self.FLAGS.gamma,
                self.FLAGS.max_episode_length-1,
                r_name='test',
                reward_weight=self.FLAGS.reward_weight,
                penalty_weight=self.FLAGS.penalty_weight
            )
        elif self.FLAGS.reward_type == 'IoU':
            self.reward_batch_list, self.reward_raw_batch = process_iou_to_reward(
                tf.squeeze(self.IoU_list_batch, axis=-1),
                self.penalty_list_batch,
                self.FLAGS.gamma,
                self.FLAGS.max_episode_length-1,
                r_name=None,
                reward_weight=self.FLAGS.reward_weight,
                penalty_weight=self.FLAGS.penalty_weight
            )
            
            self.reward_test_list, self.reward_raw_test = process_iou_to_reward(
                tf.squeeze(self.IoU_list_test, axis=-1),
                self.penalty_list_test,
                self.FLAGS.gamma,
                self.FLAGS.max_episode_length-1,
                r_name='test',
                reward_weight=self.FLAGS.reward_weight,
                penalty_weight=self.FLAGS.penalty_weight
            )
        else:
            raise Exception, 'undefined reward type' 

        ## create reinforce loss
        self.action_batch = collapse_dims(self.action_list_batch)
        self.indexes = tf.range(0, tf.shape(self.action_prob)[0]) * tf.shape(self.action_prob)[1] + tf.reshape(self.action_batch, [-1])
        self.responsible_action = tf.gather(tf.reshape(self.action_prob, [-1]), self.indexes)
        #print(self.action_prob.get_shape().as_list())
        #print(self.action_list_batch.get_shape().as_list())
        #print(self.action_batch.get_shape().as_list())
        #print(tf.reshape(self.action_prob, [-1]).get_shape().as_list())
        #print(self.indexes.get_shape().as_list())
        ## reward_batch node should not back propagate
        #self.reward_batch = tf.stop_gradient(tf.identity(collapse_dims(self.reward_batch_list)), name='reward_batch')
        self.reward_batch = tf.stop_gradient(collapse_dims(self.reward_batch_list), name='reward_batch')
        self.reward_batch_raw = collapse_dims(self.reward_raw_batch)
        #print(self.reward_batch_list.get_shape().as_list())
        #print(self.reward_batch.get_shape().as_list())
        #print(self.responsible_action.get_shape().as_list())
        #sys.exit()
        #self.reward_batch = collapse_dims(self.reward_batch_list)
        #debug_reward = tf.where(self.reward_batch>0, tf.ones_like(self.reward_batch),
        #    -1*tf.ones_like(self.reward_batch))
        self.critic_loss = \
            tf.reduce_mean(tf.norm(self.value_batch-self.reward_batch-self.FLAGS.gamma*self.value_next_batch)) \
            + tf.reduce_mean(tf.norm(self.value_last_batch))

        self.advantage_batch = self.reward_batch - self.value_batch
        self.loss_reinforce = -tf.reduce_mean(tf.log(tf.clip_by_value(self.responsible_action, 1e-20, 1))*self.reward_batch, name='reinforce_loss')
        self.loss_reinforce_recon = -tf.reduce_mean(tf.log(tf.clip_by_value(self.responsible_action, 1e-20, 1))*
            self.reward_batch_raw, name='reinforce_recon_loss')
        #self.loss_reinforce = -tf.reduce_mean(tf.log(tf.clip_by_value(self.responsible_action, 1e-10, 1))*self.advantage_batch, name='reinforce_loss')
        #self.loss_reinforce = -tf.reduce_mean(tf.log(tf.clip_by_value(self.responsible_action, 1e-10, 1))*debug_reward, name='reinforce_loss')
        self.loss_act_regu = tf.reduce_sum(self.responsible_action*tf.log(tf.clip_by_value(self.responsible_action, 1e-20, 1)))  

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
            self.optimizer_burnin = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_critic = tf.train.AdamOptimizer(self.learning_rate*10)
            self.optimizer_reinforce = tf.train.AdamOptimizer(self.learning_rate*self.FLAGS.reward_weight)

        #self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #self.opt_recon = self.optimizer.minimize(self.recon_loss, var_list=aggr_var+unet_var, global_step=self.counter)  
        #self.opt_reinforce = self.optimizer.minimize(self.loss_reinforce, var_list=aggr_var+dqn_var,
        #    global_step=self.counter)

        #so that we have always have something to optimize
        self.recon_loss, z = other.tfutil.noop(self.recon_loss)
        #print(self.update_ops)
        
        #self.opt_recon = self.optimizer.minimize(self.recon_loss, var_list=aggr_var+unet_var+[z])  
        #self.opt_recon_last = self.optimizer.minimize(self.recon_loss_last, var_list=aggr_var+unet_var+[z])  
        #self.opt_recon_first = self.optimizer.minimize(self.recon_loss_first, var_list=aggr_var+unet_var+[z])
        #self.opt_reinforce = self.optimizer.minimize(self.loss_reinforce, var_list=aggr_var+dqn_var)
        #self.opt_rein_recon = self.optimizer.minimize(
        #    self.recon_loss+self.loss_reinforce+self.FLAGS.reg_act*self.loss_act_regu,
        #    var_list=aggr_var+dqn_var+unet_var)
        #self.opt_rein_recon_last = self.optimizer.minimize(
        #    self.recon_loss_last+self.loss_reinforce+self.FLAGS.reg_act*self.loss_act_regu,
        #    var_list=aggr_var+dqn_var+unet_var)
        #self.opt_reinforce = self.optimizer.minimize(self.loss_reinforce, var_list=aggr_var+dqn_var)
        self.opt_recon = slim.learning.create_train_op(self.recon_loss, optimizer=self.optimizer_burnin, 
            variables_to_train=aggr_var+unet_var)
        self.opt_recon_last = slim.learning.create_train_op(self.recon_loss_last, optimizer=self.optimizer_burnin, 
            variables_to_train=aggr_var+unet_var)
        self.opt_recon_first = slim.learning.create_train_op(self.recon_loss_first, optimizer=self.optimizer_burnin, 
            variables_to_train=aggr_var+unet_var)
        #self.opt_reinforce = slim.learning.create_train_op(self.loss_reinforce, optimizer=self.optimizer,
        #    variables_to_train=aggr_var+dqn_var)
        #self.opt_reinforce = z
        self.opt_reinforce = slim.learning.create_train_op(self.loss_reinforce+self.FLAGS.reg_act*self.loss_act_regu, 
            optimizer=self.optimizer_reinforce, clip_gradient_norm=10, variables_to_train=dqn_var)
        #self.opt_reinforce = slim.learning.create_train_op(self.loss_reinforce+self.FLAGS.reg_act*self.loss_act_regu, 
        #    optimizer=self.optimizer_reinforce, variables_to_train=dqn_var)
        self.opt_critic = slim.learning.create_train_op(self.critic_loss, optimizer=self.optimizer_critic,
            variables_to_train=dqn_var)
        self.opt_rein_recon = slim.learning.create_train_op(self.recon_loss, optimizer=self.optimizer,
            variables_to_train=aggr_var+unet_var)
        self.opt_recon_unet = slim.learning.create_train_op(self.recon_loss, optimizer=self.optimizer,
            variables_to_train=unet_var)

    def _create_summary(self):
        #if self.FLAGS.is_training:
        self.summary_learning_rate = tf.summary.scalar('train/learning_rate', self.learning_rate)
        self.summary_loss_recon_train = tf.summary.scalar('train/loss_recon',
            self.recon_loss/(self.FLAGS.max_episode_length*self.FLAGS.batch_size))
        self.summary_loss_reinforce_train = tf.summary.scalar('train/loss_reinforce', self.loss_reinforce)
        self.summary_loss_act_regu_train = tf.summary.scalar('train/loss_act_regu', self.loss_act_regu)
        self.summary_reward_batch_train = tf.summary.scalar('train/reward_batch', tf.reduce_sum(self.reward_batch))
        self.summary_critic_loss_train = tf.summary.scalar('train/critic_loss', self.critic_loss)
        self.merged_train = tf.summary.merge_all()

    def _create_collections(self):
        dct_from_keys = lambda keys: {key: getattr(self, key) for key in keys}
        
        self.vox_prediction_collection = dict2obj(dct_from_keys(
            ['vox_pred_test', 'recon_loss_list_test', 'reward_raw_test', 'rotated_vox_test', 'vox_pred_test_rot']
        ))

        basic_list = [
            'unproj_grid_batch',
            'recon_loss',
            'recon_loss_list',
            'action_prob',
            'reward_batch_list',
            'reward_raw_batch',
            'loss_reinforce',
        ]
        
        if self.FLAGS.burin_opt == 0:
            burnin_list = basic_list[:] + ['opt_recon', 'critic_loss', 'opt_critic']
        elif self.FLAGS.burin_opt == 1:
            burnin_list = basic_list[:] + ['opt_recon_last','critic_loss', 'recon_loss_last', 'opt_critic']
        elif self.FLAGS.burin_opt == 2:
            burnin_list = basic_list[:] + ['opt_recon_first','critic_loss', 'recon_loss_first']
            
        train_list = basic_list[:] + [
            'loss_act_regu',
            'opt_rein_recon',
            'merged_train',
            'opt_reinforce',
            'action_list_batch',
            'IoU_list_batch'
        ]
        
        train_mvnet_list = basic_list[:] + ['opt_recon_last', 'merged_train']
        
        train_dqn_list = basic_list[:] + [
            'opt_reinforce',
            'opt_recon_last',
            'loss_act_regu',
            'merged_train'
        ]
        
        train_dqn_only_list = basic_list[:] + [
            'opt_reinforce',
            'loss_act_regu',
            'merged_train',
            'action_list_batch',
            'IoU_list_batch',
            'indexes',
            'responsible_action'
        ]

        debug_list = basic_list[:] + ['seg1_batch']

        self.burnin_collection = dict2obj(dct_from_keys(burnin_list))
        self.train_collection = dict2obj(dct_from_keys(train_list))
        self.train_mvnet_collection = dict2obj(dct_from_keys(train_mvnet_list))
        self.train_dqn_collection = dict2obj(dct_from_keys(train_dqn_list))
        self.train_dqn_only_collection = dict2obj(dct_from_keys(train_dqn_only_list))
        self.debug_collection = dict2obj(dct_from_keys(debug_list))
            
    def get_placeholders(self, include_vox, include_action, include_penalty, include_segs, train_mode):
        
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
            if include_penalty:
                placeholders.penalty = self.penalty_list_batch
            if include_segs:
                placeholders.seg1 = self.seg1_batch
                placeholders.seg2 = self.seg2_batch                

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
            if include_penalty:
                placeholders.penalty = self.penalty_list_test
            if include_segs:
                placeholders.seg1 = self.seg1_test
                placeholders.seg2 = self.seg2_test                

        return placeholders

    def construct_feed_dict(
            self,
            mvnet_inputs,
            include_vox,
            include_action,
            include_penalty,
            include_segs,
            train_mode = True):

        placeholders = self.get_placeholders(include_vox, include_action, include_penalty, include_segs, train_mode = train_mode)

        feed_dict = {self.is_training: train_mode}

        keys = ['rgb', 'invz', 'mask', 'azimuth', 'elevation']
        if include_vox:
            assert mvnet_inputs.vox is not None
            keys.append('vox')
        if include_action:
            assert mvnet_inputs.action is not None
            keys.append('action')
        if include_penalty:
            assert mvnet_inputs.penalty is not None
            keys.append('penalty')
        if include_segs:
            keys.extend(['seg1', 'seg2'])
        
            
        for key in keys:
            feed_dict[getattr(placeholders, key)] = getattr(mvnet_inputs, key)

        return feed_dict

    def run_collection_with_fd(self, obj, fd):
        dct = obj2dict(obj)
        outputs = self.sess.run(dct, feed_dict = fd)
        obj = dict2obj(outputs)
        return obj

    def select_action(self, mvnet_input, idx, is_training = False):
        
        feed_dict = self.construct_feed_dict(
            mvnet_input, include_vox = False, include_action = False, include_penalty = False, train_mode = False
        ) ## both during sampling and testing, train_mode is always False
    
        #if np.random.uniform(low=0.0, high=1.0) > epsilon:
        #    action_prob = self.sess.run([self.action_prob], feed_dict=feed_dict)
        #else:
        #    return np.random.randint(low=0, high=FLAGS.action_num)
        stuff = self.sess.run([self.action_prob_test], feed_dict=feed_dict)
        action_prob = np.squeeze(np.copy(stuff[0]))[idx]
        if is_training:  ## sampling during training
            print(action_prob)
            a_response = np.random.choice(action_prob, p=action_prob)

            a_idx = np.argmax(action_prob == a_response)
            print(a_idx)
        else:           ## testing
            print(action_prob)
            a_response = np.amax(action_prob)
            a_response = np.random.choice(action_prob, p=action_prob)

            a_idx = np.argmax(action_prob == a_response)
            print(a_idx)
        return a_idx

    def predict_vox_list(self, mvnet_input, is_training = False):

        feed_dict = self.construct_feed_dict(
            mvnet_input, include_vox = True, include_action = False, include_penalty = False, train_mode = is_training
        )
        return self.run_collection_with_fd(self.vox_prediction_collection, feed_dict)

    def run_step(self, mvnet_input, mode, is_training = True):
        '''mode is one of ['burnin', 'train'] '''
        feed_dict = self.construct_feed_dict(
            mvnet_input,
            include_vox = True,
            include_action = True,
            include_penalty = True,
            include_segs = self.FLAGS.use_segs,
            train_mode = is_training,
        )

        if mode == 'burnin':
            collection_to_run = self.burnin_collection
        elif mode == 'train':
            collection_to_run = self.train_collection
        elif mode == 'train_mv':
            collection_to_run = self.train_mvnet_collection
        elif mode == 'train_dqn':
            collection_to_run = self.train_dqn_collection
        elif mode == 'train_dqn_only':
            collection_to_run = self.train_dqn_only_collection
        elif mode == 'debug':
            collection_to_run = self.debug_collection
        else:
            raise Exception('invalid mode')

        return self.run_collection_with_fd(collection_to_run, feed_dict)

def obj2dict(obj):
    return obj.__dict__

def dict2obj(dct):
    x = lambda: None
    for key, val in dct.items():
        setattr(x, key, val)
    return x
    
class SingleInputFactory(object):
    def __init__(self, mem):
        self.mem = mem

    def make(self, azimuth, elevation, model_id, action = None, penalty=np.zeros((1,))):
        rgb, mask = self.mem.read_png_to_uint8(azimuth, elevation, model_id)
        invz = self.mem.read_invZ(azimuth, elevation, model_id)
        mask = (mask > 0.5).astype(np.float32) * (invz >= 1e-6)

        invz = invz[..., None]
        mask = mask[..., None]
        azimuth = azimuth[..., None]
        elevation = elevation[..., None]
        penalty = penalty[..., None]
        
        single_input = SingleInput(rgb, invz, mask, azimuth, elevation, action = action, penalty = penalty)
        return single_input
    
class SingleInput(object): 
    def __init__(self, rgb, invz, mask, azimuth, elevation, vox = None, action = None, penalty=0):
        self.rgb = rgb
        self.invz = invz
        self.mask = mask
        self.azimuth = azimuth
        self.elevation = elevation
        self.vox = vox
        self.action = action
        self.penalty = penalty

class ShapeProvider(object):
    def __init__(self, FLAGS, batch_size = None):
        self.BS = FLAGS.batch_size if (batch_size is None) else batch_size
        
        self.make_shape = lambda x: (self.BS, FLAGS.max_episode_length) + x

        self.rgb_shape = self.make_shape((FLAGS.resolution, FLAGS.resolution, 3))
        self.invz_shape = self.make_shape((FLAGS.resolution, FLAGS.resolution, 1))
        self.mask_shape = self.make_shape((FLAGS.resolution, FLAGS.resolution, 1))
        self.vox_shape = (self.BS, FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution)
        self.seg1_shape = self.vox_shape
        self.seg2_shape = self.vox_shape        
        
        self.azimuth_shape = self.make_shape((1,))
        self.elevation_shape = self.make_shape((1,))
        self.action_shape = (self.BS, FLAGS.max_episode_length-1, 1)
        self.penalty_shape = self.make_shape((1,))

        self.dtypes = {
            'rgb': np.float32,
            'invz': np.float32,
            'mask': np.float32,
            'vox': np.float32,
            'seg1': np.float32,
            'seg2': np.float32,            
            'azimuth': np.float32,
            'elevation': np.float32,
            'action': np.int32,
            'penalty': np.float32,
        }

    def make_np_zeros(self, dest = None, suffix = '_np'):
        if dest is None:
            dest = self
        for key in ['rgb', 'invz', 'mask', 'vox', 'seg1', 'seg2', 'azimuth', 'elevation', 'action', 'penalty']:
            arr = np.zeros(getattr(self, key+'_shape'), dtype = self.dtypes[key])
            setattr(dest, key+suffix, arr)

    def make_tf_ph(self, dest = None, suffix = '_ph'):
        if dest is None:
            dest = self
        for key in ['rgb', 'invz', 'mask', 'vox', 'seg1', 'seg2', 'azimuth', 'elevation', 'action', 'penalty']:
            ph = tf.placeholder(shape = getattr(self, key+'_shape'), dtype = self.dtypes[key])
            setattr(self, key+suffix, ph)
        
class MVInputs(object):
    def __init__(self, FLAGS, batch_size = None):

        self.FLAGS = FLAGS
        self.BS = FLAGS.batch_size if (batch_size is None) else batch_size

        self.provider = ShapeProvider(FLAGS, batch_size = batch_size)
        self.provider.make_np_zeros(dest = self, suffix = '')

    def put_voxel(self, voxel, batch_idx = 0):
        assert 0 <= batch_idx < self.BS
        self.vox[batch_idx, ...] = voxel

    def put_segs(self, seg1, seg2, batch_idx = 0):
        assert 0 <= batch_idx < self.BS
        self.seg1[batch_idx, ...] = seg1
        self.seg2[batch_idx, ...] = seg2

    def put(self, single_mvinput, episode_idx, batch_idx = 0):
        assert 0 <= batch_idx < self.BS
        assert 0 <= episode_idx < self.FLAGS.max_episode_length

        keys = ['rgb', 'invz', 'mask', 'azimuth', 'elevation']
        if hasattr(single_mvinput, 'action') and getattr(single_mvinput, 'action') is not None:
            keys.append('action')
        if hasattr(single_mvinput, 'penalty') and getattr(single_mvinput, 'penalty') is not None:
            keys.append('penalty')
            
        for key in keys:
            arr = getattr(self, key)
            arr[batch_idx, episode_idx, ...] = getattr(single_mvinput, key)
