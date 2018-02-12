# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Perspective Transformer Layer Implementation.

Transform the volume based on 4 x 4 perspective projection matrix.

Reference:
(1) "Perspective Transformer Nets: Perspective Transformer Nets:
Learning Single-View 3D Object Reconstruction without 3D Supervision."
Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo, Honglak Lee. In NIPS 2016
https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf

(2) Official implementation in Torch: https://github.com/xcyan/ptnbhwd

(3) 2D Transformer implementation in TF:
github.com/tensorflow/models/tree/master/research/transformer

"""

import tfpy
import tensorflow as tf
import camera
import constants as const
import numpy as np
import tfutil

def transformer(voxels,
                theta,
                out_size,
                z_near,
                z_far,
                name='PerspectiveTransformer',
                do_project = True):
    """Perspective Transformer Layer.

    Args:
        voxels: A tensor of size [num_batch, depth, height, width, num_channels].
            It is the output of a deconv/upsampling conv network (tf.float32).
        theta: A tensor of size [num_batch, 16].
            It is the inverse camera transformation matrix (tf.float32).
        out_size: A tuple representing the size of output of
            transformer layer (float).
        z_near: A number representing the near clipping plane (float).
        z_far: A number representing the far clipping plane (float).

    Returns:
        A transformed tensor (tf.float32).

    """
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([
                    n_repeats,
                ])), 1), [1, 0])
            rep = tf.to_int32(rep)
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, z, out_size):
        """Bilinear interploation layer.

        Args:
            im: A 5D tensor of size [num_batch, depth, height, width, num_channels].
                It is the input volume for the transformation layer (tf.float32).
            x: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for x (tf.float32).
            y: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for y (tf.float32).
            z: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for z (tf.float32).
            out_size: A tuple representing the output size of transformation layer
                (float).

        Returns:
            A transformed tensor (tf.float32).

        """
        with tf.variable_scope('_interpolate'):
            num_batch = im.get_shape().as_list()[0]
            depth = im.get_shape().as_list()[1]
            height = im.get_shape().as_list()[2]
            width = im.get_shape().as_list()[3]
            channels = im.get_shape().as_list()[4]

            x = tf.to_float(x)
            y = tf.to_float(y)
            z = tf.to_float(z)
            depth_f = tf.to_float(depth)
            height_f = tf.to_float(height)
            width_f = tf.to_float(width)
            # Number of disparity interpolated.
            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]
            zero = tf.zeros([], dtype='int32')
            # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
            max_z = tf.to_int32(tf.shape(im)[1] - 1)
            max_y = tf.to_int32(tf.shape(im)[2] - 1)
            max_x = tf.to_int32(tf.shape(im)[3] - 1)

            # Converts scale indices from [-1, 1] to [0, width/height/depth].
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0
            z = (z + 1.0) * (depth_f) / 2.0

            #x = tfpy.summarize_tensor(x, 'x') #(-21, 150)
            #z = tfpy.summarize_tensor(z, 'z') #(0, 128)

            x0 = tf.to_int32(tf.floor(x))
            x1 = x0 + 1
            y0 = tf.to_int32(tf.floor(y))
            y1 = y0 + 1
            z0 = tf.to_int32(tf.floor(z))
            z1 = z0 + 1

            x0_clip = tf.clip_by_value(x0, zero, max_x)
            x1_clip = tf.clip_by_value(x1, zero, max_x)
            y0_clip = tf.clip_by_value(y0, zero, max_y)
            y1_clip = tf.clip_by_value(y1, zero, max_y)
            z0_clip = tf.clip_by_value(z0, zero, max_z)
            z1_clip = tf.clip_by_value(z1, zero, max_z)
            dim3 = width
            dim2 = width * height
            dim1 = width * height * depth

            #repeat can only be run on cpu
            #base = _repeat(
            #    tf.range(num_batch) * dim1, out_depth * out_height * out_width)
            base = tf.constant(
                np.concatenate([np.array([i] * out_depth * out_height * out_width)
                                for i in range(const.BS)]).astype(np.int32)
            )
            #only works for bs = 1
            #base = tf.zeros((out_depth * out_height * out_width), dtype=tf.int32)

            base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
            base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
            base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
            base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

            idx_z0_y0_x0 = base_z0_y0 + x0_clip
            idx_z0_y0_x1 = base_z0_y0 + x1_clip
            idx_z0_y1_x0 = base_z0_y1 + x0_clip
            idx_z0_y1_x1 = base_z0_y1 + x1_clip
            idx_z1_y0_x0 = base_z1_y0 + x0_clip
            idx_z1_y0_x1 = base_z1_y0 + x1_clip
            idx_z1_y1_x0 = base_z1_y1 + x0_clip
            idx_z1_y1_x1 = base_z1_y1 + x1_clip

            # Use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.to_float(im_flat)
            i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
            i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
            i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
            i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
            i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
            i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
            i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
            i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)

            # Finally calculate interpolated values.
            x0_f = tf.to_float(x0)
            x1_f = tf.to_float(x1)
            y0_f = tf.to_float(y0)
            y1_f = tf.to_float(y1)
            z0_f = tf.to_float(z0)
            z1_f = tf.to_float(z1)
            # Check the out-of-boundary case.
            x0_valid = tf.to_float(
                tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
            x1_valid = tf.to_float(
                tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
            y0_valid = tf.to_float(
                tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
            y1_valid = tf.to_float(
                tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
            z0_valid = tf.to_float(
                tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
            z1_valid = tf.to_float(
                tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))

            w_z0_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                         (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                        1)
            w_z0_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                         (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                        1)
            w_z0_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                         (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                        1)
            w_z0_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                         (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                        1)
            w_z1_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                         (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                        1)
            w_z1_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                         (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                        1)
            w_z1_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                         (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                        1)
            w_z1_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                         (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                        1)

            output = tf.add_n([
                w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1,
                w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1,
                w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1,
                w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1
            ])
            return output

    def _meshgrid(depth, height, width, z_near, z_far):
        with tf.variable_scope('_meshgrid'):
            x_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
                [depth, height, width])
            y_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
                [depth, width, height])
            y_t = tf.transpose(y_t, [0, 2, 1])
            sample_grid = tf.tile(
                tf.linspace(float(z_near), float(z_far), depth), [width * height])
            z_t = tf.reshape(sample_grid, [height, width, depth])
            z_t = tf.transpose(z_t, [2, 0, 1])

            z_t = 1 / z_t
            d_t = 1 / z_t
            x_t /= z_t
            y_t /= z_t

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            d_t_flat = tf.reshape(d_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
            return grid

    def _noproj_meshgrid(depth, height, width, z_near, z_far):
        with tf.variable_scope('_meshgrid'):
            x_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
                [depth, height, width])
            y_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
                [depth, width, height])
            y_t = tf.transpose(y_t, [0, 2, 1])
            
            sample_grid = tf.tile(
                tf.linspace(float(z_near), float(z_far), depth), [width * height])
            z_t = tf.reshape(sample_grid, [height, width, depth])
            z_t = tf.transpose(z_t, [2, 0, 1])

            z_t = 1 / z_t
            d_t = 1 / z_t

            #originally: X = x*z/fx
            #x_t /= z_t 
            #y_t /= z_t 

            #new change
            x_t *= const.focal_length
            y_t *= const.focal_length

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            d_t_flat = tf.reshape(d_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
            return grid

    def _invproj_meshgrid(depth, height, width, z_near, z_far):
        with tf.variable_scope('_meshgrid'):
            x_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
                [depth, height, width])
            y_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
                [depth, width, height])
            y_t = tf.transpose(y_t, [0, 2, 1])
            
            sample_grid = tf.tile(
                tf.linspace(float(z_near), float(z_far), depth), [width * height])
            z_t = tf.reshape(sample_grid, [height, width, depth])
            z_t = tf.transpose(z_t, [2, 0, 1])

            z_t = 1 / z_t
            d_t = 1 / z_t

            #originally: X = x*z/fx
            #x_t /= z_t 
            #y_t /= z_t 

            #new change
            x_t *= z_t 
            y_t *= z_t 

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            d_t_flat = tf.reshape(d_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
            return grid        

    def _transform(theta, input_dim, out_size, z_near, z_far):
        with tf.variable_scope('_transform'):
            num_batch = input_dim.get_shape().as_list()[0]
            num_channels = input_dim.get_shape().as_list()[4]
            theta = tf.reshape(theta, (-1, 4, 4))
            theta = tf.cast(theta, 'float32')

            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]

            if do_project is True:
                grid = _meshgrid(out_depth, out_height, out_width, z_near, z_far)
            elif do_project == 'invert':
                grid = _invproj_meshgrid(out_depth, out_height, out_width, z_near, z_far)
            else:
                grid = _noproj_meshgrid(out_depth, out_height, out_width, z_near, z_far)
            
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))

            #grid = tfpy.summarize_tensor(grid, 'grid')

            def printgrid(grid_):
                #z in 3, 5
                #x/y in 5, -5
                zs = grid_[:, 0, :]
                print '==='
                print zs.shape
                print np.mean(zs)
                print np.max(zs)
                print np.min(zs)

            #grid = tfpy.inject_callback(grid, printgrid)

            # Transform A x (x_t', y_t', 1, d_t)^T -> (x_s, y_s, z_s, 1).
            t_g = tf.matmul(theta, grid)

            #z_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
            #y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
            #x_s = tf.slice(t_g, [0, 2, 0], [-1, 1, -1])
            #this gives a different shape, but it'll be reshaped anyway
            z_s = t_g[:, 0, :]
            y_s = t_g[:, 1, :]
            x_s = t_g[:, 2, :]

            #z_s = tfpy.summarize_tensor(z_s, 'z_s') #-1, 1
            #y_s = tfpy.summarize_tensor(y_s, 'y_s') #-1.34, 1.34
            #x_s = tfpy.summarize_tensor(x_s, 'x_s')

            z_s_flat = tf.reshape(z_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            x_s_flat = tf.reshape(x_s, [-1])

            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, z_s_flat,
                                             out_size)

            output = tf.reshape(
                input_transformed,
                tf.stack([num_batch, out_depth, out_height, out_width, num_channels]))

            return output

    with tf.variable_scope(name):
        with tf.device('/gpu:0'):
            output = _transform(theta, voxels, out_size, z_near, z_far)
        return output

def get_transform_matrix_tf(theta, phi, invert_rot = False, invert_focal = False):
    if isinstance(theta, list):
        theta = tf.stack(theta)
    if isinstance(phi, list):
        phi = tf.stack(phi)
        
    return tf.map_fn(
        lambda (t, p): get_transform_matrix_tf_(t, p, invert_rot, invert_focal),
        [theta, phi],
        parallel_iterations = 1000,
        dtype = tf.float32
    )
    
def get_transform_matrix_tf_(theta, phi, invert_rot = False, invert_focal = False):
    #INPUT IN DEGREES
    
    #extrinsic matrix:
    #
    # RRRD
    # RRRD
    # RRRD
    # 000D

    sin_phi = tf.sin(phi / 180 * np.pi)
    cos_phi = tf.cos(phi / 180 * np.pi)
    sin_theta = tf.sin(-theta / 180.0 * np.pi) #why is theta negative???
    cos_theta = tf.cos(-theta / 180.0 * np.pi)

    rotation_azimuth_flat = [
        cos_theta, 0.0, -sin_theta,
        0.0, 1.0, 0.0,
        sin_theta, 0.0, cos_theta
    ]

    rotation_elevation_flat = [
        cos_phi, sin_phi, 0.0,
        -sin_phi, cos_phi, 0.0,
        0.0, 0.0, 1.0
    ]

    f = lambda x: tf.reshape(tf.stack(x), (3, 3))
    rotation_azimuth = f(rotation_azimuth_flat)
    rotation_elevation = f(rotation_elevation_flat)

    rotation_matrix = tf.matmul(rotation_azimuth, rotation_elevation)
    if invert_rot:
        rotation_matrix = tf.linalg.inv(rotation_matrix)

    displacement = np.zeros((3, 1), dtype=np.float32)
    displacement[0, 0] = 4.0 #because the target has distance 4
    displacement = tf.constant(displacement, dtype = np.float32)
    displacement = tf.matmul(rotation_matrix, displacement)

    bottom_row = np.zeros((1, 4), dtype = np.float32)
    bottom_row[0,3] = 1.0
    bottom_row = tf.constant(bottom_row)

    #print rotation_matrix
    #print bottom_row
    #print displacement
    
    extrinsic_matrix = tf.concat([
        tf.concat([rotation_matrix, -displacement], axis = 1),
        bottom_row
    ], axis = 0)
    
    if invert_focal:
        intrinsic_diag = [1.0, float(const.focal_length), float(const.focal_length), 1.0]
    else:
        intrinsic_diag = [1.0, 1.0/float(const.focal_length), 1.0/float(const.focal_length), 1.0]
    intrinsic_matrix = tf.diag(tf.constant(intrinsic_diag, dtype = tf.float32))
    
    camera_matrix = tf.matmul(extrinsic_matrix, intrinsic_matrix)
    return camera_matrix
    
    
def get_transform_matrix(theta, phi = None, invert_rot = False, invert_focal = False):
    """Get the 4x4 Perspective Transfromation matrix used for PTN."""

    if phi is None:
        phi = const.PHI_IDX * 10.0

    #extrinsic x intrinsic
    camera_matrix = np.zeros((4, 4), dtype=np.float32)

    intrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix = np.eye(4, dtype=np.float32)

    sin_phi = np.sin(float(phi) / 180.0 * np.pi)
    cos_phi = np.cos(float(phi) / 180.0 * np.pi)
    sin_theta = np.sin(float(-theta) / 180.0 * np.pi)
    cos_theta = np.cos(float(-theta) / 180.0 * np.pi)

    #theta rotation
    rotation_azimuth = np.zeros((3, 3), dtype=np.float32)
    rotation_azimuth[0, 0] = cos_theta
    rotation_azimuth[2, 2] = cos_theta
    rotation_azimuth[0, 2] = -sin_theta
    rotation_azimuth[2, 0] = sin_theta
    rotation_azimuth[1, 1] = 1.0

    #phi rotation
    rotation_elevation = np.zeros((3, 3), dtype=np.float32)
    rotation_elevation[0, 0] = cos_phi
    rotation_elevation[0, 1] = sin_phi
    rotation_elevation[1, 0] = -sin_phi
    rotation_elevation[1, 1] = cos_phi
    rotation_elevation[2, 2] = 1.0

    #rotate phi, then theta
    rotation_matrix = np.matmul(rotation_azimuth, rotation_elevation)
    if invert_rot:
        rotation_matrix = np.linalg.inv(rotation_matrix)

    displacement = np.zeros((3, 1), dtype=np.float32)
    displacement[0, 0] = 4.0
    displacement = np.matmul(rotation_matrix, displacement)

    #assembling 4x4 from R + T
    extrinsic_matrix[0:3, 0:3] = rotation_matrix
    extrinsic_matrix[0:3, 3:4] = -displacement

    if invert_focal:
        intrinsic_matrix[2, 2] = float(const.focal_length)
        intrinsic_matrix[1, 1] = float(const.focal_length)
    else:
        intrinsic_matrix[2, 2] = 1.0 / float(const.focal_length)
        intrinsic_matrix[1, 1] = 1.0 / float(const.focal_length)

    camera_matrix = np.matmul(extrinsic_matrix, intrinsic_matrix)
    return camera_matrix

def rotate_and_project_voxel(voxel, rotmat):
    
    r = tfutil.rank(voxel)
    assert r in [4,5]
    if r == 4:
        voxel = tf.expand_dims(voxel, axis = 4)

    voxel = transformer_preprocess(voxel)
        
    with tf.device('/gpu:0'):
        out = transformer(
            voxel,
            tf.reshape(rotmat, (const.BS, 16)),
            (const.S, const.S, const.S),
            3.0,
            5.0,
        )

    out = transformer_postprocess(out)
        
    if r == 4:
        out = tf.squeeze(out, axis = 4)

    return out

def rotate_voxel(voxel, rotmat):
    with tf.device('/gpu:0'):
        return transformer(
            voxel,
            tf.reshape(rotmat, (const.BS, 16)),
            (const.S, const.S, const.S),
            3.0,
            5.0,
            do_project = False,
        )

def project_voxel(voxel):
    with tf.device('/gpu:0'):    
        rotmat = tf.constant(get_transform_matrix(0.0, 0.0), dtype = tf.float32)
        rotmat = tf.reshape(rotmat, (1, 4, 4))
        rotmat = tf.tile(rotmat, (const.BS, 1, 1))
        return transformer(
            voxel,
            tf.reshape(rotmat, (const.BS, 16)),
            (const.S, const.S, const.S),
            3.0,
            5.0,
            do_project = True,
        )

def unproject_voxel(voxel):
    #need to rewrite this

    #how does projection work?
    #input: voxels 128^3, output: voxels 128^3
    
    #first generate the meshgrid: (-1,1) for x and y, (3,5) for z
    #since X = xz/fx, next multiply x by z in the meshgrid, giving xz
    #finally, apply projectino matrix, which adds on the 1/fx factor

    #projection matrix also subtracts displacement, so we now have:
    #X = xz/fx, Y = yz/fy
    #Z = z-4
    #all these coordinates should be approximately within (-1,1)^3
    #convert to 0, 128, and tf gather from input grid to get output grid

    #####

    #now here is the strategy for unprojection.
    #start meshgrid with (-1,1) for X and Y, and (3,5) for z
    #since x = Xfx/Z, next divide X by z giving us X/z (by passing 'invert' to get_transform matrix
    #apply unprojectino matrix, which adds on fx factor (by passing 'invert' to transformer)

    #now, everything after this step should be the same. 
    
    rotmat = tf.constant(get_transform_matrix(0.0, 0.0, invert_focal = True), dtype = tf.float32)
    rotmat = tf.reshape(rotmat, (1, 4, 4))
    rotmat = tf.tile(rotmat, (const.BS, 1, 1))
    
    voxel =  transformer(
        voxel,
        tf.reshape(rotmat, (const.BS, 16)),
        (const.S, const.S, const.S),
        3.0,
        5.0,
        do_project = 'invert',
    )

    voxel = tf.reverse(voxel, axis=[2, 3])
    return voxel

def unproject_image(img_):
    #step 1: convert to voxel
    #step 2: call unproject_voxel

    def unflatten(img):
        voxel = tf.expand_dims(img, 1)
        #now BS x 1 x H x W x C
        voxel = tf.tile(voxel, (1, const.S, 1, 1, 1))
        #BS x S x S x S x C hopefully
        return voxel
    
    projected_voxel = unflatten(img_)
    return unproject_voxel(projected_voxel)

def transformer_preprocess(voxel):
    return tf.reverse(voxel, axis=[1]) #z axis

def transformer_postprocess(voxel):
    with tf.device('/gpu:0'):
        voxel = tfutil.norm01(voxel)
        voxel = tf.reverse(voxel, axis=[2, 3])
        #zxy -> xyz i think
        voxel = tf.transpose(voxel, (0, 2, 3, 1, 4))
        return voxel

def voxel2mask_aligned(voxel):
    return tf.reduce_max(voxel, axis=3)


def voxel2depth_aligned(voxel):
    voxel = tf.squeeze(voxel, axis=4)

    costgrid = tf.cast(tf.tile(
        tf.reshape(tf.range(0, const.S), (1, 1, 1, const.S)),
        (const.BS, const.S, const.S, 1)
    ), tf.float32)

    invalid = 1000 * tf.cast(voxel < 0.5, dtype=tf.float32)
    invalid_mask = tf.tile(tf.reshape(tf.constant([1.0] * (const.S - 1) + [0.0], tf.float32),
                                      (1, 1, 1, const.S)),
                           (const.BS, const.S, const.S, 1))

    costgrid = costgrid + invalid * invalid_mask

    depth = tf.expand_dims(tf.argmin(costgrid, axis=3), axis=3)

    #convert back to (3,5)
    depth = tf.cast(depth, tf.float32)
    depth += 0.5 #0.5 to 127.5
    depth /= const.S #almost 0.0 to 1.0
    depth *= 2 #almost 0.0 to 2.0
    depth += 3.0 #about 3.0 to 5.0

    return depth

def loss_against_constraint_l2(constraint, voxels):
    return tf.reduce_mean(constraint * voxels)    

def loss_against_constraint_ce(constraint, voxels):
    #we want voxels to be 1, so that the log tends to 0
    #0 so that smaller voxel -> bigger error
    return tf.reduce_mean(-tf.log(voxels + const.eps) * constraint)
