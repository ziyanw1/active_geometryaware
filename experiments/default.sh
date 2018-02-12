#!/usr/bin/env bash

python2 ./train_ae_rgb2d_persp_lmdb.py --LOG_DIR=log --CHECKPOINT_DIR=ckpt --force_delete --voxel_resolution=32 --batch_size=2 --vis_every_step=1
