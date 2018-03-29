import tfutil
import utils
import tfpy
import tensorflow as tf
import voxel
import constants as const
import numpy as np

def make_rotation_object(az0, el0, az1, el1):
    #returns object which can be used to rotate view 1 -> 0
    #see vs/nets.translate_views for reference implementation

    dtheta = az1 - az0

    r1 = voxel.get_transform_matrix(theta = 0.0, phi = -el1)
    r2 = voxel.get_transform_matrix(theta = dtheta, phi = el0)
    
    return (r1, r2)

def stack_rotation_objects(rots):
    r1s, r2s = zip(*rots)
    r1 = np.stack(r1s, axis = 0)
    r2 = np.stack(r2s, axis = 0)
    return r1, r2

def rotate_to_first(az, el):
    #input: (BS x K, BS x K)
    
    #1 is the views axis
    azs = tf.unstack(az, axis = 1)
    els = tf.unstack(el, axis = 1)
    az0 = azs[0]
    el0 = els[0]

    r1s = []
    r2s = []
    for (az1, el1) in zip(azs, els):
        dtheta = az1-az0
        r1 = voxel.get_transform_matrix_tf(theta = tf.zeros_like(el1), phi = -el1)
        r2 = voxel.get_transform_matrix_tf(theta = dtheta, phi = el0)

        r1s.append(r1)
        r2s.append(r2)
    return zip(r1s, r2s)
    
def unproject_and_rotate(depth, mask, additional, rotation = None):

    #order of concate is important
    inputs = tf.concat([mask, depth - const.DIST_TO_CAM, additional], axis = 3)
    inputs = tf.image.resize_images(inputs, (const.S, const.S))

    unprojected = unproject(inputs)

    if rotation is not None:
        rotated = voxel.rotate_voxel(unprojected, rotation[0])
        rotated = voxel.rotate_voxel(rotated, rotation[1])
    else:
        rotated = unprojected

    return rotated

def unproject(inputs):

    inputs = tf.image.resize_images(inputs, (const.S, const.S))
    #now unproject, to get our starting point
    inputs = voxel.unproject_image(inputs)

    #in addition, add on a z-map, and a local bias
    #copied from components.py
    meshgridz = tf.range(const.S, dtype = tf.float32)
    meshgridz = tf.reshape(meshgridz, (1, const.S, 1, 1))
    meshgridz = tf.tile(meshgridz, tf.stack([tfutil.batchdim(inputs), 1, const.S, const.S]))
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

def flatten(voxels):
    res = const.RESOLUTION
    
    pred_depth = voxel.voxel2depth_aligned(voxels)
    pred_mask = voxel.voxel2mask_aligned(voxels)
        
    #replace bg with grey
    hard_mask = tf.cast(pred_mask > 0.5, tf.float32)
    pred_depth *= hard_mask

    pred_depth += const.DIST_TO_CAM * (1.0 - hard_mask)

    pred_depth = tf.image.resize_images(pred_depth, (res, res))
    pred_mask = tf.image.resize_images(pred_mask, (res, res))
    return pred_depth, pred_mask

def project_and_postprocess(x):
    return voxel.transformer_postprocess(voxel.project_voxel(x))
