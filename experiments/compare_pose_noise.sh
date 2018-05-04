#!/usr/bin/env bash

cd ..

if [ $1 = "nonoise" ]; then
    #to run without pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=1 --burn_in_iter=1 \
           --test_every_step=1 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=100 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/projects/katefgroup/ziyan/models/MVnet_coef1_chair/pretrain_model.ckpt-20000'
elif [ $1 = "noise" ]; then
    #to run with pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=1 --burn_in_iter=1 \
           --test_every_step=1 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=100 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/projects/katefgroup/ziyan/models/MVnet_coef1_chair/pretrain_model.ckpt-20000' --pose_noise=True
elif [ $1 = "compoundnoise" ]; then
    #to run with pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=1 --burn_in_iter=1 \
           --test_every_step=1 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=100 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/projects/katefgroup/ziyan/models/MVnet_coef1_chair/pretrain_model.ckpt-20000' --pose_noise=True --compound_noise=True
elif [ $1 = "trainnoise" ]; then
    #to run with pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn_train' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=10 --burn_in_iter=10000 \
           --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/projects/katefgroup/ziyan/models/MVnet_coef1_chair/pretrain_model.ckpt-20000' --pose_noise=True
elif [ $1 = "trainednoise" ]; then
    #to run with pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=1 --burn_in_iter=1 \
           --test_every_step=1 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=100 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/home/ricsonc/MVnet_active/log_agent/gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn_train/log_agent/pretrain_model.ckpt-5000' --pose_noise=True
elif [ $1 = "trainedcompoundnoise" ]; then
    #to run with pose noise
    python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn' \
           --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=1 --burn_in_iter=1 \
           --test_every_step=1 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --if_bn=False \
           --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
           --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=100 --GBL_thread=True --reward_weight=10 \
           --delta=20.0 --reg_act=0.01 --penalty_weight=0 --gamma=0 --burin_opt=1 \
           --pretrain_restore=True --pretrain_restore_path='/home/ricsonc/MVnet_active/log_agent/gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn_train/log_agent/pretrain_model.ckpt-5000' --pose_noise=True
else
    echo "bad option"
fi

#results, std 5 deg

# no noise:
# eval_r_mean is 0.261275
# eval_iou_mean is 0.740392815678
# eval_loss_mean is 0.338296
# eval_r_stderr is 0.0144877180457
# eval_iou_stderr is 0.0149456286405
# eval_loss_stderr is 0.0150980904698

# noise untrained:
# eval_r_mean is 0.18219
# eval_iou_mean is 0.661401423273
# eval_loss_mean is 0.318304
# eval_r_stderr is 0.0156818524003
# eval_iou_stderr is 0.0144115319003
# eval_loss_stderr is 0.0140038698912

# compound noise untrained:
# eval_r_mean is 0.19029
# eval_iou_mean is 0.665699443237
# eval_loss_mean is 0.325491
# eval_r_stderr is 0.0170849978924
# eval_iou_stderr is 0.0156887923647
# eval_loss_stderr is 0.0145197331905

# noise trained
# eval_r_mean is 0.116682
# eval_iou_mean is 0.691341224428
# eval_loss_mean is 0.36442
# eval_r_stderr is 0.00840495154262
# eval_iou_stderr is 0.0134626389226
# eval_loss_stderr is 0.0167508423328

# compound noise trained
# eval_r_mean is 0.152769
# eval_iou_mean is 0.694944413191
# eval_loss_mean is 0.347578
# eval_r_stderr is 0.00935443341732
# eval_iou_stderr is 0.0134132378606
# eval_loss_stderr is 0.0159652382135

#test results...
# eval_r_mean is 0.261275
# eval_iou_mean is 0.740392815678
# eval_loss_mean is 0.338296
# eval_r_stderr is 0.0144877180457
# eval_iou_stderr is 0.0149456286405
# eval_loss_stderr is 0.0150980904698
