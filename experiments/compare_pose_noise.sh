#!/usr/bin/env bash

cd ..

if false; then
    #to run without pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=1 --burn_in_iter=1 \
           --test_every_step=1 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=100 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/projects/katefgroup/ziyan/models/MVnet_coef1_chair/pretrain_model.ckpt-20000'
else
    #to run with pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=1 --burn_in_iter=1 \
           --test_every_step=1 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=100 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/projects/katefgroup/ziyan/models/MVnet_coef1_chair/pretrain_model.ckpt-20000' --pose_noise=True
fi


#results...

# no noise:
# eval_r_mean is 0.261275
# eval_iou_mean is 0.740392815678
# eval_loss_mean is 0.338296
# eval_r_stderr is 0.0144877180457
# eval_iou_stderr is 0.0149456286405
# eval_loss_stderr is 0.0150980904698

# noise:
# eval_r_mean is 0.216814
# eval_iou_mean is 0.696294928955
# eval_loss_mean is 0.330079
# eval_r_stderr is 0.015002258122
# eval_iou_stderr is 0.0151807704292
# eval_loss_stderr is 0.0146419867873

#compound noise:
# eval_r_mean is 0.229544
# eval_iou_mean is 0.706733913773
# eval_loss_mean is 0.334633
# eval_r_stderr is 0.0152346983552
# eval_iou_stderr is 0.01528980891
# eval_loss_stderr is 0.0148866087198
