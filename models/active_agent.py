import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import util
from utils import tf_util

from env_data.replay_memory import ReplayMemory
from env_data.shapenet_env import ShapeNetEnv, trajectData  

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1*x + f2 * abs(x)

class ActiveAgent(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        #self.senv = ShapeNetEnv(FLAGS)
        #self.replay_mem = ReplayMemory(FLAGS)

        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.activation_fn = lrelu
        self.rgb_batch = tf.placeholder(dtype=tf.float32, 
            shape=[None, FLAGS.resolution, FLAGS.resolution, 3], name='rgb_batch')
        self.vox_batch = tf.placeholder(dtype=tf.float32,
            shape=[None, FLAGS.voxel_resolution, FLAGS.voxel_resolution, FLAGS.voxel_resolution],
            name='vox_batch')
        self.action_batch = tf.placeholder(dtype=tf.int32, shape=[None, ], name='action_batch')
        self.reward_batch = tf.placeholder(dtype=tf.float32, shape=[None, ], name='reward_batch')
        
        #print('========== Starting burning in memories')
        #self.burn_in()
        #print('========== Burn in done. {} trajectories burnt in'.format(self.FLAGS.burn_in_length))

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
                net_rgb = tf.flatten(net_rgb, scope='rgb_flatten')

                net_vox = slim.conv3d(vox, 64, kernel_size=[3,3], stride=[1,1], padding='SAME', scope='vox_conv1')
                net_vox = slim.conv3d(net_vox, 128, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='vox_conv2')
                net_vox = slim.conv3d(net_vox, 256, kernel_size=[3,3], stride=[1,1], padding='SAME', scope='vox_conv3')
                net_vox = slim.conv3d(net_vox, 256, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='vox_conv4')
                net_vox = slim.conv3d(net_vox, 512, kernel_size=[3,3], stride=[2,2], padding='SAME', scope='vox_conv5')
                net_vox = slim.flatten(net_vox, scope='vox_flatten')
                
                net_feat = tf.concat([net_rgb, net_vox], axis=1)
                net_feat = slim.fully_connected(net_feat, 4096, scope='fc6')
                net_feat = slim.fully_connected(net_feat, 4096, scope='fc7')
                logits = slim.fully_connected(net_feat, self.FLAGS.action_num, activation_fn=None, scope='fc8')

                return tf.nn.softmax(logits), logits

    def _create_policy_net(self):

        self.rgb_batch_norm = tf.subtract(tf.div(self.rgb_batch, 255.), 0.5)
        self.action_prob, self.logits = self._create_dqn_two_stream(self.rgb_batch_norm, self.vox_batch)
    
    def _create_loss(self):
        self.indexes = tf.range(0, tf.shape(self.action_prob)[0]) * tf.shape(self.action_prob)[1] + self.action_batch
        self.responsible_action = tf.gather(tf.reshape(self.action_prob, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_action)*self.reward_batch, name='reinforce_loss')

    #def burn_in(self): 

    #    FLAGS = self.FLAGS
    #    
    #    state = self.senv.reset(True)
    #    actions = []
    #    for i in range(FLAGS.max_episode_length*FLAGS.burn_in_length):
    #        actions.append(np.random.randint(0,8))
    #        state, next_state, done, model = self.senv.step(actions[-1])
    #        if done:
    #            traj_state = state
    #            traj_state[0] += [next_state[0]]
    #            traj_state[1] += [next_state[1]]
    #            temp_traj = trajectData(traj_state, actions, model)
    #            #print temp_traj.states, temp_traj.actions, temp_traj.model_id
    #            self.replay_mem.append(temp_traj)
    #            self.senv.reset(True)
    #            actions = []


    #def select_action(self):
    #    pass

    #def train(self):
    #    pass

    #def test(self):
    #    pass
