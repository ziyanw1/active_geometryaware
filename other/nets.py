import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as const
import tfpy
import voxel

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

def unproject(inputs):

    bs = int(list(inputs.get_shape())[0])
    
    inputs = tf.image.resize_images(inputs, (const.S, const.S))
    #now unproject, to get our starting point
    inputs = voxel.unproject_image(inputs)

    #in addition, add on a z-map, and a local bias
    #copied from components.py
    meshgridz = tf.range(const.S, dtype = tf.float32)
    meshgridz = tf.reshape(meshgridz, (1, const.S, 1, 1))
    meshgridz = tf.tile(meshgridz, (bs, 1, const.S, const.S))
    meshgridz = tf.expand_dims(meshgridz, axis = 4) 
    meshgridz = (meshgridz + 0.5) / (const.S/2.0) - 1.0 #now (-1,1)

    #get the rough outline
    unprojected_mask = tf.expand_dims(inputs[:,:,:,:,0], 4)
    unprojected_depth = tf.expand_dims(inputs[:,:,:,:,1], 4)
    outline_thickness = 0.1
    outline = tf.cast(tf.logical_and(
        unprojected_depth <= meshgridz,
        unprojected_depth + 0.1 > meshgridz
    ), tf.float32)
    outline *= unprojected_mask

    if const.DEBUG_UNPROJECT:
        #return tf.expand_dims(inputs[:,:,:,:,0], 4) #this is the unprojected mask
        return outline

    if const.USE_LOCAL_BIAS:
        bias = tf.get_variable("voxelnet_bias", dtype=tf.float32,
                               shape = [1, const.S, const.S, const.S, 1],
                               initializer=tf.zeros_initializer())
        bias = tf.tile(bias, (bs, 1, 1, 1, 1))


    inputs_ = [inputs, meshgridz]
    if const.USE_LOCAL_BIAS:
        inputs_.append(bias)
    if const.USE_OUTLINE:
        inputs_.append(outline)
    inputs = tf.concat(inputs_, axis = 4)
    return inputs

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
