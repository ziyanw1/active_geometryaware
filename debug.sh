#!/usr/bin/env bash

python train_active_mvnet.py --is_training=True --task_name='debug_' \
     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-3 --burn_in_length=1 \
     --test_every_step=1 --save_every_step=10 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
     --resolution=128 --category='03001627' --mem_length=100 --burn_in_iter=1 --if_save_eval=True \
     --unet_name='OUTLINE' --agg_name='OUTLINE'
