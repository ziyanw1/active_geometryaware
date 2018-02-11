import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
import numpy as np
import math
import sys
import os

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
        self.data_loader = data_loader(self.FLAGS)

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        self.counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.test_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.activation_fn = lrelu

        self.global_i = tf.Variable(0, name='global_i', trainable=False)
        self.set_i_to_pl = tf.placeholder(tf.int32,shape=[], name='set_i_to_pl')
        self.assign_i_op = tf.assign(self.global_i, self.set_i_to_pl)

        # Add ops to save and restore all variable 
        self.saver = tf.train.Saver()
        
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
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
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        # create summary
        self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.LOG_DIR, 'train'), self.sess.graph)

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
                net_out_depth = slim.conv2d_transpose(net_up1_, out_channel, kernel_size=[4,4], stride=[2,2], padding='SAME', \
                    activation_fn=tf.tanh, normalizer_fn=None, normalizer_params=None, scope='unet_out')

            
        return net_out_depth, net_bottleneck

    def _create_network(self):
        with tf.device('/gpu:0'):

            # TODO: load data
            #self.data_loader = DataLoader(self.FLAGS)
            self.rgb_batch = self.data_loader.rgb_batch
            self.invZ_batch = self.data_loader.invZ_batch
            self.rgb_batch_norm = tf.subtract(tf.div(self.rgb_batch, 255.), 0.5)

            #self.rgb_batch = tf.placeholder(tf.float32, shape=(None,128,128,3), name='input_rgb')
            #self.invZ_batch = tf.placeholder(tf.float32, shape=(None,128,128,1), name='gt_invZ')
            
            self.invZ_pred, self.z_rgb = self._create_unet(self.rgb_batch_norm, trainable=True, if_bn=True, scope_name='unet_rgb2depth') 


    def _create_loss(self):
        self.depth_recon_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.invZ_pred-self.invZ_batch),
            [1,2,3])), 0)

        self.loss = self.depth_recon_loss


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
        self.optimizer = self.optimizer.minimize(self.loss, var_list=unet_vars, global_step=self.counter)

    def _create_summary(self):

        self.summary_learning_rate = tf.summary.scalar('train/learning_rate', self.learning_rate)
        self.summary_loss_train = tf.summary.scalar('train/loss', self.loss)
        self.summary_loss_test = tf.summary.scalar('test/loss', self.loss)

        self.summary_loss_depth_recon_train = tf.summary.scalar('train/loss_depth_recon', self.depth_recon_loss)
        self.summary_loss_depth_recon_test = tf.summary.scalar('test/loss_depth_recon', self.depth_recon_loss)

        self.summary_train = [self.summary_loss_train, self.summary_loss_depth_recon_train, self.summary_learning_rate]
        self.summary_test = [self.summary_loss_test, self.summary_loss_depth_recon_test]

        self.merge_train = tf.summary.merge(self.summary_train)
        self.merge_test = tf.summary.merge(self.summary_test)
