import numpy as np
import random
import tensorflow as tf
from tensorpack import *
import math
import tflearn
import scipy
import scipy.io as sio
import time
from tensorflow.python.framework import ops
import warnings
import os

import threading

class GeneratorRunner(object):
    "Custom runner that that runs an generator in a thread and enqueues the outputs."
    def __init__(self, generator, placeholders, enqueue_op, close_op):
        self._generator = generator
        self._placeholders = placeholders
        self._enqueue_op = enqueue_op
        self._close_op = close_op
    def _run(self, sess, coord):
        try:
            while not coord.should_stop():
                try:
                    # print "======== values = self._generator.get_data()"
                    values = next(self._generator)
                    # print values.shape
                    # values = [values]
                    if len(values) != len(self._placeholders):
                        print "======== len(values), len(self._placeholders)", len(values), len(self._placeholders)
                    assert len(values) == len(self._placeholders), \
                        'generator values and placeholders must have the same length'

                    #if len(values[0]) == self._placeholders[0].get_shape().as_list()[0]:
                    feed_dict = {placeholder: value \
                        for placeholder, value in zip(self._placeholders, values)}
                    sess.run(self._enqueue_op, feed_dict=feed_dict)
                except (StopIteration, tf.errors.OutOfRangeError):
                    try:
                        sess.run(self._close_op)
                    except Exception:
                        pass
                    return
        except Exception as ex:
            if coord:
                coord.request_stop(ex)
            else:
                raise

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        "Called by `start_queue_runners`."
        print "===== GeneratorRunner.create_threads"
        thread = threading.Thread(
            target=self._run,
            args=(sess, coord))
        if coord:
            coord.register_thread(thread)
        if daemon:
            thread.daemon = True
        if start:
            thread.start()
        return [thread]

def read_batch_generator(
        generator, dtypes, shapes, batch_size,
        queue_capacity=1000,
        allow_smaller_final_batch=True):
    "Reads values from an generator, queues, and batches."
    assert len(dtypes) == len(shapes), 'dtypes and shapes must have the same length'
    queue = tf.FIFOQueue(
        capacity=queue_capacity,
        dtypes=dtypes,
        shapes=shapes)
    placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
    print placeholders
    enqueue_op = queue.enqueue(placeholders)
    close_op = queue.close(cancel_pending_enqueues=False)
    queue_runner = GeneratorRunner(generator, placeholders, enqueue_op, close_op)
    tf.train.add_queue_runner(queue_runner)
    if allow_smaller_final_batch:
        return queue.dequeue_up_to(batch_size)
    else:
        print "===== returning read_batch_generator->queue.dequeue_many"
        return queue.dequeue_many(batch_size)

class data_loader(object):
    def __init__(self, flags):
        ## All variables ##
        global FLAGS
        FLAGS = flags
        self.out_size = (FLAGS.num_point, 3)
        self.resolution = FLAGS.resolution
        self.vox_reso = FLAGS.voxel_resolution
        self.is_training = tf.placeholder(dtype=bool,shape=[],name='gen-is_training')

        # data_lmdb_path = "/home/rz1/Documents/Research/3dv2017_PBA/data/lmdb"
        # data_lmdb_path = "/data_tmp/lmdb/"
        # data_lmdb_path = "/newfoundland/rz1/lmdb/"
        #data_lmdb_path = "./data/lmdb/"

        data_lmdb_path = flags.data_path
        data_lmdb_train_file = flags.data_file + '_train.tfr'
        data_lmdb_test_file = flags.data_file + '_test.tfr'
        
        # data_lmdb_path = "/home/ziyan/3dv2017_PBA_out/data/lmdb/"
        # self.data_pcd_train = data_lmdb_path + "randLampbb8Full_%s_%d_train_imageAndShape.lmdb"%(FLAGS.cat_name, FLAGS.num_point)
        # self.data_pcd_train = data_lmdb_path + "random_randomLamp0822_%s_%d_train_imageAndShape_single.lmdb"%(FLAGS.cat_name, FLAGS.num_point)
        self.data_ae_train = os.path.join(data_lmdb_path, data_lmdb_train_file)
        #self.data_pcd_train = data_lmdb_path + "random_randLamp1005_%s_%d_train_imageAndShape_single_persp.amdb"%(FLAGS.cat_name, FLAGS.num_point)
        # self.data_pcd_train = '/data_tmp/lmdb/badRenderbb9_car_24576_train_imageAndShape.lmdb'
        # self.data_pcd_test = data_lmdb_path + "random_randomLamp0822_%s_%d_test_imageAndShape_single.lmdb"%(FLAGS.cat_name, FLAGS.num_point)
        self.data_ae_test = os.path.join(data_lmdb_path, data_lmdb_test_file)
        #self.data_pcd_test = data_lmdb_path + "random_randLamp1005_%s_%d_test_imageAndShape_single_persp.lmdb"%(FLAGS.cat_name, FLAGS.num_point)
        # self.data_pcd_test = '/newfoundland/rz1/lmdb/badRenderbb9_car_24576_test_imageAndShape.lmdb'
        
        buffer_size = 32
        parall_num = 16
        self.batch_size = FLAGS.batch_size # models used in a batch

        '''
        self.ds_train = LMDBData(self.data_ae_train, shuffle=True) #[pcd, axis_angle_single, tw_single, angle_single, rgb_single, style]
        self.x_size_train = self.ds_train.size()
        self.ds_train = LocallyShuffleData(self.ds_train, buffer_size)
        self.ds_train = PrefetchData(self.ds_train, buffer_size, parall_num)
        self.ds_train = LMDBDataPoint(self.ds_train)
        self.ds_train = PrefetchDataZMQ(self.ds_train, parall_num)
        self.ds_train = BatchData(self.ds_train, self.batch_size, remainder=False, use_list=True) # no smaller tail batch
        self.ds_train = RepeatedData(self.ds_train, -1)  # -1 for repeat infinite times
        # TestDataSpeed(self.ds_train).start_test() # 164.15it/s
        self.ds_train.reset_state()
        '''

        raise Exception, 'update size'
        self.ds_train = TFRecordData(self.data_ae_train, size = 6) #[pcd, axis_angle_single, tw_single, angle_single, rgb_single, style]
        self.x_size_train = self.ds_train.size()
        self.ds_train = LocallyShuffleData(self.ds_train, buffer_size)
        self.ds_train = PrefetchData(self.ds_train, buffer_size, parall_num)
        self.ds_train = PrefetchDataZMQ(self.ds_train, parall_num)
        #self.ds_train = RepeatedData(self.ds_train, 10) #remove this later
        self.ds_train = BatchData(self.ds_train, self.batch_size, remainder=False, use_list=True) # no smaller tail batch
        self.ds_train = RepeatedData(self.ds_train, -1)  # -1 for repeat infinite times
        # TestDataSpeed(self.ds_train).start_test() # 164.15it/s
        self.ds_train.reset_state()

        '''
        #self.ds_test = LMDBData(self.data_pcd_test, shuffle=True) #[pcd, axis_angle_single, tw_single, angle_single, rgb_single, style]
        self.ds_test = LMDBData(self.data_ae_test, shuffle=False) #[pcd, axis_angle_single, tw_single, angle_single, rgb_single, style]
        self.x_size_test = self.ds_test.size()
        #self.ds_test = LocallyShuffleData(self.ds_test, 200)
        self.ds_test = PrefetchData(ds=self.ds_test, nr_prefetch=buffer_size, nr_proc=parall_num)
        self.ds_test = LMDBDataPoint(self.ds_test)
        self.ds_test = PrefetchDataZMQ(ds=self.ds_test, nr_proc=parall_num)
        # all dataset will be iterated
        self.ds_test = BatchData(self.ds_test, self.batch_size, remainder=False, use_list=True)
        self.ds_test = RepeatedData(self.ds_test, -1)
        # TestDataSpeed(self.ds_test).start_test()
        self.ds_test.reset_state()
        '''

        raise Exception, 'update size'        
        self.ds_test = TFRecordData(self.data_ae_test, size=3) #[pcd, axis_angle_single, tw_single, angle_single, rgb_single, style]
        self.x_size_test = self.ds_test.size()
        self.ds_test = PrefetchData(ds=self.ds_test, nr_prefetch=buffer_size, nr_proc=parall_num)
        self.ds_test = PrefetchDataZMQ(ds=self.ds_test, nr_proc=parall_num)
        # all dataset will be iterated
        #self.ds_test = RepeatedData(self.ds_test, 10)
        self.ds_test = BatchData(self.ds_test, self.batch_size, remainder=False, use_list=True)
        self.ds_test = RepeatedData(self.ds_test, -1)
        self.ds_test.reset_state()

        self.rgb_batch_train, self.invZ_batch_train, self.mask_batch_train, self.sn_batch_train,\
            self.angles_batch_train, self.vox_batch_train = read_batch_generator\
            (generator=self.ds_train.get_data(), dtypes=[tf.uint8, tf.float32, tf.float32, tf.float32, tf.float32,\
                tf.uint8], \
                shapes=[[self.batch_size, self.resolution, self.resolution, 3], [self.batch_size, self.resolution, \
                self.resolution, 1], [self.batch_size, self.resolution, self.resolution, 1], \
                [self.batch_size, self.resolution, self.resolution, 3],\
                [self.batch_size, 3], [self.batch_size, self.vox_reso, self.vox_reso, self.vox_reso]], batch_size=1, queue_capacity=100)

        self.rgb_batch_test, self.invZ_batch_test, self.mask_batch_test, self.sn_batch_test,\
            self.angles_batch_test, self.vox_batch_test = read_batch_generator\
            (generator=self.ds_test.get_data(), dtypes=[tf.uint8, tf.float32, tf.float32, tf.float32, tf.float32,
                tf.uint8], \
                shapes=[[self.batch_size, self.resolution, self.resolution, 3], [self.batch_size, self.resolution, \
                self.resolution, 1], [self.batch_size, self.resolution, self.resolution, 1], \
                [self.batch_size, self.resolution, self.resolution, 3],\
                [self.batch_size, 3], [self.batch_size, self.vox_reso, self.vox_reso, self.vox_reso]], batch_size=1, queue_capacity=100)

        #self.rgb_batch_test, self.invZ_batch_test, self.mask_batch_test, self.sn_batch_test,\
        #    self.angles_batch_test, self.vox_batch_test = read_batch_generator\
        #    (generator=self.ds_test.get_data(), dtypes=[tf.uint8, tf.float32, tf.float32, tf.float32, tf.float32,
        #        tf.uint8], \
        #        shapes=[[1, self.resolution, self.resolution, 3], [1, self.resolution, \
        #        self.resolution, 1], [1, self.resolution, self.resolution, 1], \
        #        [1, self.resolution, self.resolution, 3],\
        #        [1, 3], [1, self.vox_reso, self.vox_reso, self.vox_reso]], batch_size=self.batch_size, queue_capacity=100)

        self.rgb_batch = tf.reshape(tf.cond(self.is_training, \
            lambda: tf.to_float(self.rgb_batch_train), \
            lambda: tf.to_float(self.rgb_batch_test)), [self.batch_size, self.resolution, self.resolution, 3])
        ## normalization happens in autoencoder
        
        self.invZ_batch = tf.reshape(tf.cond(self.is_training, \
            lambda: tf.to_float(self.invZ_batch_train), \
            lambda: tf.to_float(self.invZ_batch_test)), [self.batch_size, self.resolution, self.resolution, 1])
        
        self.mask_batch = tf.reshape(tf.cond(self.is_training, \
            lambda: tf.to_float(self.mask_batch_train), \
            lambda: tf.to_float(self.mask_batch_test)), [self.batch_size, self.resolution, self.resolution, 1])

        self.sn_batch = tf.reshape(tf.cond(self.is_training, \
            lambda: tf.to_float(self.sn_batch_train), \
            lambda: tf.to_float(self.sn_batch_test)), [self.batch_size, self.resolution, self.resolution, 3])
        
        self.angles_batch = tf.reshape(tf.cond(self.is_training, \
            lambda: tf.to_float(self.angles_batch_train), \
            lambda: tf.to_float(self.angles_batch_test)), [self.batch_size, 3])

        self.voxel_batch = tf.reshape(tf.cond(self.is_training, \
            lambda: tf.to_float(self.vox_batch_train), \
            lambda: tf.to_float(self.vox_batch_test)), [self.batch_size, self.vox_reso, self.vox_reso, self.vox_reso])

