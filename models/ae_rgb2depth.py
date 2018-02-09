import tensorflow as tf
import tflearn
import numpy as np
import math
import sys
import os

import tensorflow.contrib.layers as ly
from utils import util
from utils import tf_util

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = o.5 * (1-leak)
        return f1*x + f2 * abs(x)


class AE_rgb2d(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        self.counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.test_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.activation_fn = lrelu


    def _create_unet(self, rgb, trainable=True, if_bn=False, reuse=False, scope_name='rgb2depth'):
        

        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            if if_bn:
                batch_normalizer_gen = slim.batch_norm
                batch_norm_params_gen = {'is_training': self.is_training, 'decay': self.FLAGS.bn_decay}

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
                net_up5__ = tf.concat([net_up5, net_down5], axis=-1)

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
                net_out_depth = slim.conv3d_transpose(net_up1_, 1, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    activation_fn=tf.tanh, normalizer_fn=None, normalizer_params=None, scope='unet_out')

            
        return net_out_depth, net_bottleneck

    def _create_network(self):
        with tf.device('/gpu:0'):

            ## TODO: load data
            self.data_loader = DataLoader(self.FLAGS)
            self.rgb_batch = self.data_loader.rgb_batch
            self.depth_batch = self.data_loader.depth_batch
            
            self.depth_pred, self.z_rgb = self._create_unet(self.rgb_batch,trainable=True) 


    def _create_loss(self):
        self.depth_recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.depth_pred-self.depth_batch), [1,2]), 0)

        self.loss = self.depth_recon_loss


    def _create_optimizer(self):
        
        unet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARBLES, scope='unet')

        if self.FLAGS.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.FLAGS.momentum)
        elif self.FLAGS.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = self.optimizer.minimize(self.loss, var_list=unet_vars, global_step=self.counter)
