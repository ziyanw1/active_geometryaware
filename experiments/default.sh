#!/usr/bin/env bash

python2 ./train_ae_rgb2d_persp_lmdb.py --LOG_DIR=log --CHECKPOINT_DIR=ckpt --force_delete --voxel_resolution=128 --batch_size=1 --vis_every_step=1 --data_path=data/lmdb128 --data_file=rgb2depth_single_train_0212.lmdb --loss_name=voxel_loss --opt_step_name=opt_3d_step
