#!/usr/bin/env bash

python ./train_ae_rgb2d_persp_lmdb.py --LOG_DIR=debug --CHECKPOINT_DIR=ckpt --task_name=rgb2depth_0213 --force_delete --voxel_resolution=32 \
    --batch_size=16 --vis_every_step=1 --data_path=data/lmdb --data_file=rgb2depth_single_0212 \
    --save_every_step=10000 --loss_name=sketch_loss --opt_step_name=opt_step --use_gan=True 

#python ./train_ae_rgb2d_persp_lmdb.py --LOG_DIR=log --CHECKPOINT_DIR=ckpt --task_name=rgb2depth_0218_unet --force_delete --voxel_resolution=32 \
#    --batch_size=16 --vis_every_step=1 --data_path=data/lmdb --data_file=rgb2depth_single_0212 --network_name='unet'\
#    --save_every_step=10000 --loss_name=sketch_loss --opt_step_name=opt_step --max_iter=400000 

#python ./train_ae_rgb2d_nopara_lmdb.py --LOG_DIR=log --CHECKPOINT_DIR=ckpt --task_name=rgb2depth_0213 --force_delete --voxel_resolution=32 \
#    --batch_size=16 --vis_every_step=1 --data_path=data/lmdb --data_file=rgb2depth_single_0212 \
#    --save_every_step=10000 --loss_name=sketch_loss --opt_step_name=opt_step 
