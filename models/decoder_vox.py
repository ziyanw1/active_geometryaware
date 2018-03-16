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

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1*x + f2 * abs(x)


class Decoder_vox(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.z_vox = tf.placeholder(tf.float32, shape=[1, 2, 2, 2, 512], name='z_vox')

        self.counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.test_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.activation_fn = lrelu

        self.global_i = tf.Variable(0, name='global_i', trainable=False)
        self.set_i_to_pl = tf.placeholder(tf.int32,shape=[], name='set_i_to_pl')
        self.assign_i_op = tf.assign(self.global_i, self.set_i_to_pl)

        
        self._create_network()
        
        # Add ops to save and restore all variable 
        self.saver = tf.train.Saver(max_to_keep=5)
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
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        # create summary
        self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.LOG_DIR, 'train'), self.sess.graph)
        
    def _create_unet(self, rgb, out_channel=1, trainable=True, if_bn=False, reuse=False, scope_name='unet_3d'):

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
    
    def _create_encoder(self, vox, trainable=True, if_bn=False, reuse=False, scope_name='ae_encoder'):

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

                net_down1 = slim.conv3d(vox, 64, kernel_size=3, stride=1, padding='SAME', scope='ae_conv1')
                net_down2 = slim.conv3d(net_down1, 128, kernel_size=4, stride=2, padding='SAME', scope='ae_conv2')
                net_down3 = slim.conv3d(net_down2, 256, kernel_size=3, stride=1, padding='SAME', scope='ae_conv3')
                net_down4 = slim.conv3d(net_down3, 512, kernel_size=4, stride=2, padding='SAME', scope='ae_conv4')
                net_down5 = slim.conv3d(net_down4, 512, kernel_size=4, stride=2, padding='SAME', scope='ae_conv5')
                net_bottleneck = slim.conv3d(net_down5, 512, kernel_size=4, stride=2, padding='SAME', scope='ae_conv6')

        return net_bottleneck
    
    def _create_decoder(self, z_rgb, trainable=True, if_bn=False, reuse=False, scope_name='ae_decoder'):

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

                net_up5 = slim.conv3d_transpose(z_rgb, 512, kernel_size=[4,4,4], stride=[2,2,2], padding='SAME', \
                    scope='ae_deconv6')
                net_up4 = slim.conv3d_transpose(net_up5, 512, kernel_size=[4,4,4], stride=[2,2,2], padding='SAME', \
                    scope='ae_deconv5')
                net_up3 = slim.conv3d_transpose(net_up4, 512, kernel_size=[4,4,4], stride=[2,2,2], padding='SAME', \
                    scope='ae_deconv4')
                net_up2 = slim.conv3d_transpose(net_up3, 256, kernel_size=[3,3,3], stride=[1,1,1], padding='SAME', \
                    scope='ae_deconv3')
                net_up1 = slim.conv3d_transpose(net_up2, 128, kernel_size=[4,4,4], stride=[2,2,2], padding='SAME', \
                    scope='ae_deconv2')
                net_up0 = slim.conv3d_transpose(net_up1, 64, kernel_size=[3,3,3], stride=[1,1,1], padding='SAME', \
                    scope='ae_deconv1')
                net_out_ = slim.conv3d_transpose(net_up0, 1, kernel_size=[3,3,3], stride=[1,1,1], padding='SAME', \
                    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='ae_out')

        return tf.nn.sigmoid(net_out_), net_out_
            
    def _create_discriminator(self, preds, trainable=True, if_bn=False, reuse=False, scope_name='discriminator'):
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

                net = slim.conv3d(preds, 64, kernel_size=[4, 4, 4], stride=[2,2,2], padding='SAME',
                    scope='discriminator_conv1') 
                net = slim.conv3d(net, 128, kernel_size=[4, 4, 4], stride=[2,2,2], padding='SAME',
                    scope='discriminator_conv2')
                net = slim.conv3d(net, 256, kernel_size=[4, 4, 4], stride=[2,2,2], padding='SAME',
                    scope='discriminator_conv3')
                net = slim.conv3d(net, 256, kernel_size=[3, 3, 3], stride=[1,1,1], padding='SAME',
                    scope='discriminator_conv4')
                net = slim.conv3d(net, 512, kernel_size=[3, 3, 3], stride=[1,1,1], padding='SAME',
                    scope='discriminator_conv5')

                net = slim.flatten(net, scope='discriminator_flatten')
                net = slim.fully_connected(net, 4096, scope='discriminator_fc6')
                net = slim.dropout(net, scope='discriminator_dropout1')
                net = slim.fully_connected(net, 4096, scope='discriminator_fc7')
                net = slim.dropout(net, scope='discriminator_dropout2')
                net = slim.fully_connected(net, 1, activation_fn=None, scope='discriminator_fc8')
                return tf.nn.sigmoid(net), net

    def _create_network(self):
        with tf.device('/gpu:0'):

            # TODO: load data
            
            if self.FLAGS.network_name == 'ae':
                self.preds, self.pred_logits = self._create_decoder(self.z_vox, trainable=True, if_bn=True, scope_name='vox_ae_decoder')
            #elif self.FLAGS.network_name == 'unet':
            #    self.preds, self.z_rgb = self._create_unet(self.rgb_batch_norm, out_channel=5, trainable=True,
            #        if_bn=True, scope_name='unet_rgb2depth') 
            else:
                raise NotImplementedError

    #def _create_loss(self):

    #    #self.recon_loss = tf.losses.sigmoid_cross_entropy(self.voxel_batch, self.pred_logits)

    #    #if self.FLAGS.use_gan:
    #    #    self.D_loss = -tf.reduce_mean(tf.log(self.real_pred) + tf.log(1.0-self.fake_pred))
    #    #    self.G_loss = -tf.reduce_mean(tf.log(self.fake_pred))

    #def _create_optimizer(self):
    #    
    #    vox_ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vox_ae')
    #    if self.FLAGS.if_constantLr:
    #        self.learning_rate = self.FLAGS.learning_rate
    #        #self._log_string(tf_util.toGreen('===== Using constant lr!'))
    #    else:  
    #        self.learning_rate = get_learning_rate(self.counter, self.FLAGS)

    #    if self.FLAGS.optimizer == 'momentum':
    #        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.FLAGS.momentum)
    #        if self.FLAGS.use_gan:
    #            self.optimizer_D = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.FLAGS.momentum)
    #    elif self.FLAGS.optimizer == 'adam':
    #        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    #        if self.FLAGS.use_gan:
    #            self.optimizer_D = tf.train.AdamOptimizer(self.learning_rate)
    #        

    #    if self.FLAGS.use_gan:
    #        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vox_discriminator')
    #        self.opt_D = self.optimizer_D.minimize(
    #            0.1*self.D_loss, var_list=D_vars, global_step=self.counter
    #        )
    #        self.opt_step = self.optimizer.minimize(
    #            self.recon_loss+0.1*self.G_loss, var_list=vox_ae_vars, global_step=self.counter
    #        )
    #    else:
    #        self.opt_step = self.optimizer.minimize(
    #            self.recon_loss, var_list=vox_ae_vars, global_step=self.counter
    #        )
        

    #def _create_summary(self):

    #    self.summary_learning_rate = tf.summary.scalar('train/learning_rate', self.learning_rate)
    #    self.summary_loss_train = tf.summary.scalar('train/loss', self.recon_loss)
    #    self.summary_loss_test = tf.summary.scalar('test/loss', self.recon_loss)

    #    self.summary_train = [self.summary_loss_train, self.summary_learning_rate]
    #    self.summary_test = [self.summary_loss_test]

    #    if self.FLAGS.use_gan:
    #        self.summary_Dloss_train = tf.summary.scalar('train/D_loss', self.D_loss)
    #        self.summary_Gloss_train = tf.summary.scalar('train/G_loss', self.G_loss)
    #        self.summary_train += [self.summary_Dloss_train, self.summary_Gloss_train]

    #    self.merge_train = tf.summary.merge(self.summary_train)
    #    self.merge_test = tf.summary.merge(self.summary_test)
