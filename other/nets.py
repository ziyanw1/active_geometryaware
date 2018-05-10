import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as const
import tfpy
import voxel
import sys

from lsm.ops import convgru, convlstm, collapse_dims, uncollapse_dims 
from tensorflow import summary as summ

def encoder_decoder(inputs, pred_dim, blocks, base_channels=16, aux_input=None, bn = True):
    feat_stack = []
    pred_stack = []
    h = const.H
    w = const.W

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding="SAME",
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm if bn else None,
                        normalizer_params={'is_training': True, #const.mode != 'test',
                                           'epsilon': 1e-5,
                                           'decay': 0.9,
                                           'scale': True,
                                           'updates_collections': None},
                        stride=1,
                        weights_initializer=tf.truncated_normal_initializer(stddev=1E-4),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        # ENCODER
        net = inputs
        chans = base_channels
        # first, one conv at full res
        net = slim.conv2d(net, chans, [3, 3], stride=1,
                          scope='conv%d' % 0)
        feat_stack.append(net)
        for i in range(blocks):
            chans *= 2
            net = slim.conv2d(net, chans, [3, 3], stride=2,
                              scope='conv%d_1' % (i + 1))
            net = slim.conv2d(net, chans, [3, 3], stride=1,
                              scope='conv%d_2' % (i + 1))
            if i != blocks - 1:
                feat_stack.append(net)
            h /= 2
            w /= 2

        if aux_input:
            # aux input must be a vector
            _, aux_d = aux_input.get_shape()
            aux_block = tf.reshape(aux_input, (const.BS, 1, 1, aux_d))
            aux_block = tf.tile(aux_block, [1, h, w, 1])
            net = tf.concat([net, aux_block], axis=3)
            # 1x1 conv the aux inputs
            net = slim.conv2d(net, chans, [1, 1], activation_fn=None, normalizer_fn=None)

        # DECODER
        for i in reversed(range(blocks)):
            # predict from these feats
            pred = slim.conv2d(net, pred_dim, [3, 3], stride=1,
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='pred%d' % (i + 1))
            pred_stack.append(pred)
            # deconv the feats
            chans = chans / 2
            h *= 2
            w *= 2

            net = tf.image.resize_nearest_neighbor(net, [h, w], name="upsamp%d" % (i + 1))
            net = slim.conv2d(net, chans, [3, 3], stride=1, scope='conv%d' % (i + 1))
            # concat [upsampled pred, deconv, saved conv from earlier]
            __vals = [tf.image.resize_images(pred, [h, w]), net, feat_stack.pop()]
            net = tf.concat(axis=3, values=__vals, name="concat%d" % (i + 1))

        # one last pred at full res
        pred = slim.conv2d(net, pred_dim, [3, 3], stride=1,
                           activation_fn=None,
                           normalizer_fn=None,
                           scope='pred%d' % 0)
        pred_stack.append(pred)

    assert not feat_stack
    # return pred_stack
    return pred


def encoder_multidecoder(inputs, pred_dims, blocks, base_channels=16,
                         aux_input=None, freeze_encoder=False, nin_first=False,
                         bn = True):
    feat_stack = []
    pred_stack = []
    h = const.H
    w = const.W

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding="SAME",
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm if bn else None,
                        normalizer_params={'is_training': True, #const.mode != 'test', #NO
                                           'epsilon': 1e-5,
                                           'scale': True,
                                           'updates_collections': None},
                        stride=1,
                        weights_initializer=tf.truncated_normal_initializer(stddev=1E-4),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        # ENCODER
        net = inputs
        chans = base_channels
        # first, one conv at full res
        do_nin = nin_first or inputs.get_shape()[-1] > 64  # do nin here
        if do_nin and not nin_first:
            print 'WARNING -- doing nin'
        first_kernel_size = [1, 1] if nin_first else [3, 3]
        
        net = slim.conv2d(net, chans, first_kernel_size, stride=1,
                          scope='conv%d' % 0)
        feat_stack.append(net)
        for i in range(blocks):
            chans *= 2
            net = slim.conv2d(net, chans, [3, 3], stride=2,
                              scope='conv%d_1' % (i + 1))
            net = slim.conv2d(net, chans, [3, 3], stride=1,
                              scope='conv%d_2' % (i + 1))
            if i != blocks - 1:
                feat_stack.append(net)
            h /= 2
            w /= 2

        if aux_input:
            # aux input must be a vector
            _, aux_d = aux_input.get_shape()
            aux_block = tf.reshape(aux_input, (const.BS, 1, 1, aux_d))
            aux_block = tf.tile(aux_block, [1, h, w, 1])
            net = tf.concat([net, aux_block], axis=3)
            # 1x1 conv the aux inputs
            net = slim.conv2d(net, chans, [1, 1], activation_fn=None, normalizer_fn=None)

        if freeze_encoder:
            net = tf.stop_gradient(net)
            feat_stack = map(tf.stop_gradient, feat_stack)

        # DECODERS
        net = [net for i in pred_dims]
        for i in reversed(range(blocks)):
            current_feats = feat_stack.pop()
            chans = chans / 2
            h *= 2
            w *= 2

            for j, pred_dim in enumerate(pred_dims):
                # predict from these feats
                pred = slim.conv2d(net[j], pred_dim, [3, 3], stride=1,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='pred_%d_%d' % (i + 1, j))

                net[j] = tf.image.resize_nearest_neighbor(
                    net[j], [h, w], name="upsamp_%d_%d" % (i + 1, j)
                )

                net[j] = slim.conv2d(net[j], chans, [3, 3], stride=1,
                                     scope='conv_%d_%d' % (i + 1, j))

                # concat [upsampled pred, deconv, saved conv from earlier]
                __vals = [tf.image.resize_images(pred, [h, w]), net[j], current_feats]
                net[j] = tf.concat(axis=3, values=__vals, name="concat_%d_%d" % (i + 1, j))

        assert not feat_stack
        preds = []
        # one last pred at full res
        for j, pred_dim in enumerate(pred_dims):
            pred = slim.conv2d(net[j], pred_dim, [3, 3], stride=1,
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='pred_%d_%d' % (0, j))
            preds.append(pred)
    return preds


def voxel_net(inputs, aux = None, bn = True, outsize = 128, built_in_transform = False):
    bn_trainmode = ((const.mode != 'test') and (not const.rpvx_unsup))
    if const.force_batchnorm_trainmode:
        bn_trainmode = True
    if const.force_batchnorm_testmode:
        bn_trainmode = False
        
    normalizer_params={'is_training': bn_trainmode, 
                       'decay': 0.9,
                       'epsilon': 1e-5,
                       'scale': True,
                       'updates_collections': None}
    
    with slim.arg_scope([slim.conv2d, slim.conv3d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm if bn else None,
                        normalizer_params=normalizer_params
                        ):
        
        #the encoder part
        dims = [64, 128, 256, 512, const.VOXNET_LATENT]
        ksizes = [11, 5, 5, 5, 8]
        strides = [4, 2, 2, 2, 1]
        paddings = ['SAME'] * 4 + ['VALID']

        net = inputs
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):            
            if const.DEBUG_HISTS:
                tf.summary.histogram('encoder_%d' % i, net)
            net = slim.conv2d(net, dim, ksize, stride=stride, padding=padding)

        if aux is not None:
            aux = tf.reshape(aux, (const.BS, 1, 1, -1))
            net = tf.concat([aux, net], axis = 3)

        #two FC layers, as prescribed
        for i in range(2):
            net = slim.fully_connected(net, const.VOXNET_LATENT)

        if built_in_transform:
            tnet = net
            tnet = slim.fully_connected(tnet, 128)
            tnet = slim.fully_connected(tnet, 128)
            pose_logits = slim.fully_connected(tnet, 20, normalizer_fn = None)
            pose_ = tf.nn.softmax(pose_logits)
            
            angles = [20.0 * i for i in range(const.V)]
            rot_mats = map(utils.voxel.get_transform_matrix, angles)
            rot_mats = map(lambda x: tf.constant(x, dtype=tf.float32), rot_mats)
            rot_mats = tf.expand_dims(tf.stack(rot_mats, axis=0), axis = 0)
            pose_ = tf.reshape(pose_, (const.BS, const.V, 1, 1))
            rot_mat = tf.reduce_sum(rot_mats * pose_, axis = 1)

            print rot_mat
            
            #do some things here.. predict weights for each rotmat
            
        #net is 1 x 1 x 1 x ?
        net = tf.reshape(net, (const.BS, 1, 1, 1, -1))

        if outsize == 128:
            chans = [256, 128, 64, 32, 1]
            strides = [1] + [2] * 4
            ksizes = [8] + [4] * 5
            paddings = ['VALID'] + ['SAME'] * 4
            activation_fns = [tf.nn.relu] * 4 + [None] 

        elif outsize == 32:
            chans = [256, 128, 64, 1]
            strides = [1, 2, 2, 2]
            ksizes = [4, 2, 2, 2] 
            paddings = ['VALID'] + ['SAME'] * 3
            activation_fns = [tf.nn.relu] * 3 + [None] 

        else:
            raise Exception, 'unsupported outsize %d' % outsize


        decoder_trainable = not const.rpvx_unsup #don't ruin the decoder by FTing
        #normalizer_params_ = dict(normalizer_params.items())
        #if not decoder_trainable:
        #    normalizer_params_['is_training'] = False

        for i, (chan, stride, ksize, padding, activation_fn) \
            in enumerate(zip(chans, strides, ksizes, paddings, activation_fns)):

            if const.DEBUG_HISTS:
                tf.summary.histogram('decoder_%d' % i, net)

            #before
            if i == -1:
                net = tfpy.summarize_tensor(net, 'layer %d' % i)

            if i == len(chans)-1:
                norm_fn = None
            else:
                norm_fn = slim.batch_norm

            net = slim.conv3d_transpose(
                net, chan, ksize, stride=stride, padding=padding, activation_fn = activation_fn,
                normalizer_fn = norm_fn, trainable = decoder_trainable
            )

        if const.DEBUG_HISTS:
            tf.summary.histogram('before_sigmoid', net)

    net = tf.nn.sigmoid(net)
    if const.DEBUG_HISTS:
        tf.summary.histogram('after_sigmoid', net)

    if built_in_transform:
        net = voxel.rotate_voxel(net, rot_mat)

    return net

def voxel_net_3d(inputs, aux = None, bn = True, outsize = 128, d0 = 16):


    # B x S x S x S x 25
    ###########################

    if aux is not None:
        assert tfutil.rank(aux) == 2
        aux_dim = int(tuple(aux.get_shape())[1])
        
    #aux is used for the category input
    bn_trainmode = ((const.mode != 'test') and (not const.rpvx_unsup))
    if const.force_batchnorm_trainmode:
        bn_trainmode = True
    if const.force_batchnorm_testmode:
        bn_trainmode = False

    normalizer_params={'is_training': bn_trainmode, 
                       'decay': 0.9,
                       'epsilon': 1e-5,
                       'scale': True,
                       'updates_collections': None}

    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm if bn else None,
                        normalizer_params=normalizer_params
                        ):
        
        #the encoder part
        if const.NET3DARCH == '3x3':
            dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
            ksizes = [3, 3, 3, 3, 3]
            strides = [2, 2, 2, 2, 2]
            paddings = ['SAME'] * 5
        elif const.NET3DARCH == 'marr':
            dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
            ksizes = [4, 4, 4, 4, 8]
            strides = [2, 2, 2, 2, 1] 
            paddings = ['SAME'] * 4 + ['VALID']
        elif const.NET3DARCH == 'marr_small':
            # 32 -> 16 -> 8 -> 4 -> 1
            dims = [d0, 2*d0, 4*d0, 8*d0]
            ksizes = [4, 4, 4, 4] 
            strides = [2, 2, 2, 1] 
            paddings = ['SAME'] * 3 + ['VALID']            
        elif const.NET3DARCH == 'marr_64':
            # 64 -> 32 -> 16 -> 8 -> 4 -> 1
            dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
            ksizes = [4, 4, 4, 4, 4] 
            strides = [2, 2, 2, 2, 1]
            paddings = ['SAME'] * 4 + ['VALID']            
        else:
            raise Exception, 'unsupported network architecture'

        net = inputs
        skipcons = [net]
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim.conv3d(net, dim, ksize, stride=stride, padding=padding)

            skipcons.append(net)
        
        #BS x 4 x 4 x 4 x 256

        if aux is not None:
            aux = tf.reshape(aux, (-1, 1, 1, 1, aux_dim))

            if const.NET3DARCH == '3x3':
                aux = tf.tile(aux, (1, 4, 4, 4, 1)) #!!!!!!!!!!!!!! hardcoded value
            elif const.NET3DARCH == 'marr':
                pass #really do nothing ;)
            else:
                raise Exception, 'unsupported networka rchitecture'
                
            net = tf.concat([aux, net], axis = 4)

        #fix from here..
        if const.NET3DARCH == '3x3':
            chans = [128, 64, 32, 16, 1]
            strides = [2, 2, 2, 2, 2]
            ksizes = [3, 3, 3, 3, 3]
            paddings = ['SAME'] * 5
            activation_fns = [tf.nn.relu] * 4 + [None] #important to have the last be none
        elif const.NET3DARCH == 'marr':
            chans = [8*d0, 4*d0, 2*d0, d0, 1]
            strides = [1, 2, 2, 2, 2]
            ksizes = [8, 4, 4, 4, 4]
            paddings = ['VALID'] + ['SAME'] * 4
            activation_fns = [tf.nn.relu] * 4 + [None] #important to have the last be none
        elif const.NET3DARCH == 'marr_small':
            chans = [4*d0, 2*d0, d0, 1]
            strides = [1, 2, 2, 2]
            ksizes = [4, 4, 4, 4]
            paddings = ['VALID'] + ['SAME'] * 3
            activation_fns = [tf.nn.relu] * 3 + [None] #important to have the last be none            
        elif const.NET3DARCH == 'marr_64':
            chans = [8*d0, 4*d0, 2*d0, d0, 1]
            strides = [1, 2, 2, 2, 2]
            ksizes = [4, 4, 4, 4, 4]
            paddings = ['VALID'] + ['SAME'] * 4
            activation_fns = [tf.nn.relu] * 4 + [None] #important to have the last be none
        else:
            raise Exception, 'unsupported network architecture'

        decoder_trainable = not const.rpvx_unsup #don't ruin the decoder by FTing

        skipcons.pop() #we don't want the innermost layer as skipcon
        
        for i, (chan, stride, ksize, padding, activation_fn) \
            in enumerate(zip(chans, strides, ksizes, paddings, activation_fns)):

            if i == len(chans)-1:
                norm_fn = None
            else:
                norm_fn = slim.batch_norm

            net = slim.conv3d_transpose(
                net, chan, ksize, stride=stride, padding=padding, activation_fn = activation_fn,
                normalizer_fn = norm_fn, trainable = decoder_trainable
            )

            #now concatenate on the skip-connection
            net = tf.concat([net, skipcons.pop()], axis = 4)

        #one last 1x1 conv to get the right number of output channels
        net = slim.conv3d(
            net, 1, 1, 1, padding='SAME', activation_fn = None,
            normalizer_fn = slim.batch_norm, trainable = decoder_trainable
        )

    net = tf.nn.sigmoid(net)

    return net

#just the encoder now
def voxel_encoder(inputs, aux, reuse):

    net = inputs
    with slim.arg_scope([slim.conv3d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.layer_norm,
                        ):
        
        dims = [16, 32, 64, 128, 512]
        ksizes = [4, 4, 4, 4, 4, 4]
        strides = [2, 2, 2, 2, 2, 1] 
        paddings = ['SAME'] * 5 + ['VALID']

        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            if const.DEBUG_HISTS and not reuse:
                tf.summary.histogram('critic_conv_%d' % i, net)
            
            net = slim.conv3d(
                net, dim, ksize,
                stride=stride, padding=padding,
                scope = 'conv_%d' % i
            )

        net = tf.reshape(net, (const.BS, -1))
        net = tf.concat([aux, net], axis = 1)
        
        dims = [512, 256, 128, 1]
        acts = [tf.nn.relu]*3 + [None]
        norms = [slim.layer_norm] *3 + [None]
        for i, (dim, act, norm) in enumerate(zip(dims, acts, norms)):
            if const.DEBUG_HISTS and not reuse:
                tf.summary.histogram('critic_fc_%d' % i, net)

            net = slim.fully_connected(
                net, dim,
                activation_fn = act, normalizer_fn = norm,
                scope = 'fc_%d' % i
            )

    if const.DEBUG_HISTS and not reuse:
        tf.summary.histogram('critic_out', net)

    return net


def voxel_net_3d_v2(inputs, aux = None, bn = True, bn_trainmode = 'train',
                    freeze_decoder = False, d0 = 16, return_logits = False, return_feats = False,
                    debug = False):

    decoder_trainable = not freeze_decoder
    input_size = list(inputs.get_shape())[1]
    if input_size == 128:
        arch = 'marr128'
    elif input_size == 64:
        arch = 'marr64'
    elif input_size == 32:
        arch = 'marr32'
    else:
        raise Exception, 'input size not supported'

    if aux is not None:
        assert tfutil.rank(aux) == 2
        aux_dim = int(tuple(aux.get_shape())[1])

    normalizer_params={'is_training': bn_trainmode, 
                       'decay': 0.9,
                       'epsilon': 1e-5,
                       'scale': True,
                       'updates_collections': None,
                       'trainable': decoder_trainable}

    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm if bn else None,
                        normalizer_params=normalizer_params
                        ):
        
        if arch == 'marr128':
            # 128 -> 64 -> 32 -> 16 -> 8 -> 1
            dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
            ksizes = [4, 4, 4, 4, 8]
            strides = [2, 2, 2, 2, 1] 
            paddings = ['SAME'] * 4 + ['VALID']
        elif arch == 'marr64':
            # 64 -> 32 -> 16 -> 8 -> 4 -> 1
            dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
            ksizes = [4, 4, 4, 4, 4] 
            strides = [2, 2, 2, 2, 1]
            paddings = ['SAME'] * 4 + ['VALID']            
        elif arch == 'marr32':
            # 32 -> 16 -> 8 -> 4 -> 1
            dims = [d0, 2*d0, 4*d0, 8*d0]
            ksizes = [4, 4, 4, 4] 
            strides = [2, 2, 2, 1] 
            paddings = ['SAME'] * 3 + ['VALID']            
        #inputs = slim.batch_norm(inputs, decay=0.9, scale=True, epsilon=1e-5,
        #    updates_collections=None, is_training=bn_trainmode, trainable=decoder_trainable)

        net = inputs

        if debug:
            summ.histogram('voxel_net_3d_input', net)

        skipcons = [net]
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim.conv3d(net, dim, ksize, stride=stride, padding=padding)
            skipcons.append(net)
            if debug:
                summ.histogram('voxel_net_3d_enc_%d' % i, net)
        
        if aux is not None:
            aux = tf.reshape(aux, (-1, 1, 1, 1, aux_dim))
            net = tf.concat([aux, net], axis = 4)

        if arch == 'marr128':
            chans = [8*d0, 4*d0, 2*d0, d0, 1]
            strides = [1, 2, 2, 2, 2]
            ksizes = [8, 4, 4, 4, 4]
            paddings = ['VALID'] + ['SAME'] * 4
        elif arch == 'marr64':
            chans = [8*d0, 4*d0, 2*d0, d0, 1]
            strides = [1, 2, 2, 2, 2]
            ksizes = [4, 4, 4, 4, 4]
            paddings = ['VALID'] + ['SAME'] * 4
        elif arch == 'marr32':
            chans = [4*d0, 2*d0, d0, 1]
            strides = [1, 2, 2, 2]
            ksizes = [4, 4, 4, 4]
            paddings = ['VALID'] + ['SAME'] * 3


        skipcons.pop() #we don't want the innermost layer as skipcon
        
        for i, (chan, stride, ksize, padding) in enumerate(zip(chans, strides, ksizes, paddings)):

            net = slim.conv3d_transpose(
                net, chan, ksize, stride=stride, padding=padding, trainable=decoder_trainable
            )
            #now concatenate on the skip-connection
            net = tf.concat([net, skipcons.pop()], axis = 4)

            if net.shape[1] == 32:
                feats = net

            if debug:
                summ.histogram('voxel_net_3d_dec_%d' % i, net)
                
        #one last 1x1 conv to get the right number of output channels
        net = slim.conv3d(
            net, 1, 1, 1, padding='SAME', activation_fn = None,
            normalizer_fn = None, trainable = decoder_trainable
        )
        if debug:
            summ.histogram('voxel_net_3d_logits', net)
        

    net_ = tf.nn.sigmoid(net)
    if debug:
        summ.histogram('voxel_net_3d_output', net_)
    
    rvals = [net_]
    if return_logits:
        rvals.append(net)
    if return_feats:
        rvals.append(feats)
    if len(rvals) == 1:
        return rvals[0]
    return tuple(rvals)

#calling this 'unet_same' because it looks pretty similar to the original network i was using, except using
#same instead of valid padding on the innermost layer
def unet_same(vox_feat, channels, FLAGS, trainable=True, if_bn=False, reuse=False,
              is_training = True, activation_fn = tf.nn.relu, scope_name='unet_3d'):
    
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        if if_bn:
            batch_normalizer_gen = slim.batch_norm
            batch_norm_params_gen = {'is_training': is_training, 'decay': FLAGS.bn_decay}
        else:
            batch_normalizer_gen = None
            batch_norm_params_gen = None

        if FLAGS.if_l2Reg:
            weights_regularizer = slim.l2_regularizer(1e-5)
        else:
            weights_regularizer = None

        with slim.arg_scope([slim.fully_connected, slim.conv3d, slim.conv3d_transpose],
                activation_fn=activation_fn,
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

def unet_valid_sparese(vox_feat, mask, channels, FLAGS, trainable=True, if_bn=False, reuse=False,
              is_training = True, activation_fn = tf.nn.relu, scope_name='unet_3d'):
    
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        if if_bn:
            batch_normalizer_gen = slim.batch_norm
            batch_norm_params_gen = {'is_training': is_training, 'decay': FLAGS.bn_decay}
        else:
            batch_normalizer_gen = None
            batch_norm_params_gen = None

        if FLAGS.if_l2Reg:
            weights_regularizer = slim.l2_regularizer(1e-5)
        else:
            weights_regularizer = None

        with slim.arg_scope([slim.fully_connected, slim.conv3d, slim.conv3d_transpose],
                activation_fn=activation_fn,
                trainable=trainable,
                normalizer_fn=batch_normalizer_gen,
                normalizer_params=batch_norm_params_gen,
                weights_regularizer=weights_regularizer):
            
            mask_down1 = tf.stop_gradient(mask)
            net_down1 = slim.conv3d(vox_feat*tf.tile(mask_down1, [1,1,1,1,16]), 16, kernel_size=4, stride=2, padding='SAME', scope='unet_conv1')
            mask_down2 = tf.stop_gradient(slim.max_pool3d(mask_down1, kernel_size=4, stride=2, padding='SAME')) 
            net_down2 = slim.conv3d(net_down1*tf.tile(mask_down2, [1,1,1,1,16]) , 32, kernel_size=4, stride=2, padding='SAME', scope='unet_conv2')
            #net_down2 = slim.conv3d(net_down1 , 32, kernel_size=4, stride=2, padding='SAME', scope='unet_conv2')
            mask_down3 = tf.stop_gradient(slim.max_pool3d(mask_down2, kernel_size=4, stride=2, padding='SAME'))
            net_down3 = slim.conv3d(net_down2*tf.tile(mask_down3, [1,1,1,1,32]), 64, kernel_size=4, stride=2, padding='SAME', scope='unet_conv3')
            #net_down3 = slim.conv3d(net_down2, 64, kernel_size=4, stride=2, padding='SAME', scope='unet_conv3')
            mask_down4 = tf.stop_gradient(slim.max_pool3d(mask_down3, kernel_size=4, stride=2, padding='SAME'))
            net_down4 = slim.conv3d(net_down3*tf.tile(mask_down4, [1,1,1,1,64]), 128, kernel_size=4, stride=2, padding='SAME', scope='unet_conv4')
            #net_down4 = slim.conv3d(net_down3, 128, kernel_size=4, stride=2, padding='SAME', scope='unet_conv4')
            mask_down5 = tf.stop_gradient(slim.max_pool3d(mask_down4, kernel_size=4, stride=2, padding='SAME'))
            net_down5 = slim.conv3d(net_down4*tf.tile(mask_down5, [1,1,1,1,128]), 256, kernel_size=4, stride=2, padding='SAME', scope='unet_conv5')
            #net_down5 = slim.conv3d(net_down4, 256, kernel_size=4, stride=2, padding='SAME', scope='unet_conv5')
            mask_down6 = tf.stop_gradient(slim.max_pool3d(mask_down5, kernel_size=4, stride=2, padding='SAME'))
            
            net_up4 = slim.conv3d_transpose(net_down5*tf.tile(mask_down6, [1,1,1,1,256]), 128, kernel_size=4, stride=2, padding='SAME', \
                scope='unet_deconv4')
            #net_up4 = slim.conv3d_transpose(net_down5, 128, kernel_size=4, stride=2, padding='SAME', \
            #    scope='unet_deconv4')
            net_up4_ = tf.concat([net_up4, net_down4], axis=-1)
            net_up3 = slim.conv3d_transpose(net_up4_*tf.tile(mask_down5, [1,1,1,1,256]), 64, kernel_size=4, stride=2, padding='SAME', \
                scope='unet_deconv3')
            #net_up3 = slim.conv3d_transpose(net_up4_, 64, kernel_size=4, stride=2, padding='SAME', \
            #    scope='unet_deconv3')
            net_up3_ = tf.concat([net_up3, net_down3], axis=-1)
            net_up2 = slim.conv3d_transpose(net_up3_*tf.tile(mask_down4, [1,1,1,1,128]), 32, kernel_size=4, stride=2, padding='SAME', \
                scope='unet_deconv2')
            #net_up2 = slim.conv3d_transpose(net_up3_, 32, kernel_size=4, stride=2, padding='SAME', \
            #    scope='unet_deconv2')
            net_up2_ = tf.concat([net_up2, net_down2], axis=-1)
            net_up1 = slim.conv3d_transpose(net_up2_*tf.tile(mask_down3, [1,1,1,1,64]), 16, kernel_size=4, stride=2, padding='SAME', \
                scope='unet_deconv1')
            #net_up1 = slim.conv3d_transpose(net_up2_, 16, kernel_size=4, stride=2, padding='SAME', \
            #    scope='unet_deconv1')
            net_up1_ = tf.concat([net_up1, net_down1], axis=-1)
            #net_out_ = slim.conv3d(net_up1_, 1, kernel_size=4, stride=2, padding='SAME', \
            #    activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='unet_deconv_out')
            ## heavy load
            net_up0 = slim.conv3d_transpose(net_up1_*tf.tile(mask_down2, [1,1,1,1,32]), channels, kernel_size=4, stride=2, padding='SAME', \
                scope='unet_deconv0')
            #net_up0 = slim.conv3d_transpose(net_up1_, channels, kernel_size=4, stride=2, padding='SAME', \
            #    scope='unet_deconv0')
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


def gru_aggregator(unproj_grids, channels, FLAGS, trainable=True, if_bn=False, reuse=False,
                   is_training = True, activation_fn = tf.nn.relu, scope_name='aggr_64'):

    #unproj_grids_collap = collapse_dims(unproj_grids)
    unproj_grids_list = tf.unstack(unproj_grids, axis=1)
    
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        if if_bn:
            batch_normalizer_gen = slim.batch_norm
            batch_norm_params_gen = {'is_training': is_training, 
                                     'decay': FLAGS.bn_decay,
                                     'epsilon': 1e-5,
                                     'scale': True,
                                     'updates_collections': None,
                                     'trainable': trainable}
        else:
            batch_normalizer_gen = None
            batch_norm_params_gen = None

        if FLAGS.if_l2Reg:
            weights_regularizer = slim.l2_regularizer(1e-5)
        else:
            weights_regularizer = None

        ## create fuser
        with slim.arg_scope([slim.fully_connected, slim.conv3d],
                activation_fn=activation_fn,
                trainable=trainable,
                normalizer_fn=batch_normalizer_gen,
                normalizer_params=batch_norm_params_gen,
                weights_regularizer=weights_regularizer):

            #net_unproj = slim.conv3d(unproj_grids_collap, 16, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv1')
            net_unproj_first = slim.conv3d(unproj_grids_list[0], 16, kernel_size=1, stride=1, padding='VALID' ,
                scope='aggr_conv1_split')
            conv3d_reuse = lambda x: slim.conv3d(x, 16, kernel_size=1, stride=1, padding='VALID', reuse=True,
                scope='aggr_conv1_split')
            net_unproj_follow = tf.unstack(tf.map_fn(conv3d_reuse, tf.stack(unproj_grids_list[1:])), axis=0)
            #net_unproj = slim.conv3d(net_unproj, 64, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv2')
            #net_unproj = slim.conv3d(net_unproj, 64, kernel_size=3, stride=1, padding='SAME', scope='aggr_conv3')

            ## the input for convgru should be in shape of [bs, episode_len, vox_reso, vox_reso, vox_reso, ch]
            #net_unproj = uncollapse_dims(net_unproj, FLAGS.batch_size, FLAGS.max_episode_length)
            #net_unproj = uncollapse_dims(
            #    net_unproj,
            #    unproj_grids.get_shape().as_list()[0]/FLAGS.max_episode_length, 
            #    FLAGS.max_episode_length
            #)
            net_unproj = tf.stack([net_unproj_first]+net_unproj_follow, axis=1)
            
            net_pool_grid, _ = convgru(net_unproj, filters=channels) ## should be shape of [bs, len, vox_reso x 3, ch] 

    return net_pool_grid

def pooling_aggregator(unproj_grids, channels, FLAGS, trainable=True, reuse=False,
                       is_training = True, scope_name='aggr_64'):


    unproj_grids = collapse_dims(unproj_grids)
        
    with tf.variable_scope(scope_name, reuse = reuse) as scope:

        #a simple 1x1 convolution -- no BN
        feats = slim.conv3d(
            unproj_grids, channels, activation_fn = None, kernel_size=1, stride=1, trainable = trainable
        )

    l = FLAGS.max_episode_length
    uncollapse = lambda x: uncollapse_dims(x, x.get_shape().as_list()[0]/l, l)
            
    feats = uncollapse(feats)
    unproj_grids = uncollapse(unproj_grids)
    
    #B x E x V x V X V x C

    def fn_pool(feats, pool_fn, id_, givei = False):
        outputs = []
        base = id_
        for i in range(FLAGS.max_episode_length):
            if givei:
                base = pool_fn(feats[:,i], base, i)
            else:
                base = pool_fn(feats[:,i], base)
            outputs.append(base)
        return tf.stack(outputs, axis = 1)

    # return tf.concat([
    #     fn_pool(unproj_grids, tf.maximum, unproj_grids[:,0]),
    #     fn_pool(unproj_grids, tf.minimum, unproj_grids[:,0]),
    #     fn_pool(feats, tf.maximum, feats[:,0])
    # ], axis = -1)

    #max may be a bad idea
    #return fn_pool(feats, tf.maximum, feats[:,0])

    return fn_pool(
        feats,
        lambda x, prev, i: i/(i+1.0) * prev + 1/(i+1.0) * x,
        feats[:,0],
        givei = True
    )

def max_pooling_aggregator(unproj_grids, channels, FLAGS, trainable=True, reuse=False,
                       is_training = True, scope_name='aggr_64'):


    unproj_grids = collapse_dims(unproj_grids)
        
    with tf.variable_scope(scope_name, reuse = reuse) as scope:

        #a simple 1x1 convolution -- no BN
        feats = slim.conv3d(
            unproj_grids, channels, activation_fn = None, kernel_size=1, stride=1, trainable = trainable
        )

    l = FLAGS.max_episode_length
    uncollapse = lambda x: uncollapse_dims(x, x.get_shape().as_list()[0]/l, l)
            
    feats = uncollapse(feats)
    unproj_grids = uncollapse(unproj_grids)
    
    #B x E x V x V X V x C

    def fn_pool(feats, pool_fn, id_, givei = False):
        outputs = []
        base = id_
        for i in range(FLAGS.max_episode_length):
            if givei:
                base = pool_fn(feats[:,i], base, i)
            else:
                base = pool_fn(feats[:,i], base)
            outputs.append(base)
        return tf.stack(outputs, axis = 1)

    #return tf.concat([
    #    fn_pool(unproj_grids, tf.maximum, unproj_grids[:,0]),
    #    fn_pool(unproj_grids, tf.minimum, unproj_grids[:,0]),
    #    fn_pool(feats, tf.maximum, feats[:,0])
    #], axis = -1)

    #max may be a bad idea
    return fn_pool(feats, tf.maximum, feats[:,0])

    #return fn_pool(
    #    feats,
    #    lambda x, prev, i: i/(i+1.0) * prev + 1/(i+1.0) * x,
    #    feats[:,0],
    #    givei = True
    #)
