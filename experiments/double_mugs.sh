## GRU IoU pretrain

cd ..

# python train_active_mvnet.py --is_training=True --LOG_DIR=log_agent_all --task_name='asdf3'     \
#  --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-3 --burn_in_length=10 --burn_in_iter=20000 \
#  --test_every_step=50 --save_every_step=1000 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#   --resolution=128 --category='3333' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#  --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=1 --GBL_thread=True --reward_weight=10 \
# --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --gamma=0 --burin_opt=3 \
# --pretrain_restore=True --pretrain_restore_path='/home/ricsonc/MVnet_active/log_agent_all/asdf2/log_agent/pretrain_model.ckpt-1000' --use_segs=False --reproj_mode=True --train_filename_prefix='single' --val_filename_prefix='single' --test_filename_prefix='single' --eval0=True


# python train_active_mvnet.py --is_training=True --LOG_DIR=log_agent_all --task_name='asdf10'     \
#  --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-3 --burn_in_length=10 --burn_in_iter=10001 \
#  --test_every_step=100 --save_every_step=500 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#   --resolution=128 --category='3333' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#  --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=8 --GBL_thread=True --reward_weight=10 \
# --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --gamma=0 --burin_opt=0 \
#  --pretrain_restore=True --pretrain_restore_path='/home/ricsonc/MVnet_active/log_agent_all/asdf9/log_agent/pretrain_model.ckpt-1000' --use_segs=True --force_delete=True --seg_cluster_mode='kcenters'

#--train_filename_prefix='single' --val_filename_prefix='single' --test_filename_prefix='single'

#burin 0 for trian all

#works with asdf1, ckpt 5000

#now let's test...
python test_active_mvnet.py --LOG_DIR=log_agent_all --task_name='asdf10_test_v3'     \
 --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-3 --burn_in_length=10 --burn_in_iter=10001 \
 --test_every_step=1 --save_every_step=1000 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
  --resolution=128 --category='3333' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
 --use_coef=False --loss_coef=5 --if_save_eval=True --test_episode_num=134 --GBL_thread=True --reward_weight=10 \
--delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --gamma=0 --burin_opt=1 \
 --pretrain_restore=True --pretrain_restore_path='/home/ricsonc/MVnet_active/log_agent_all/asdf9/log_agent/pretrain_model.ckpt-1000' --use_segs=True --force_delete=True --seg_cluster_mode='kcenters'

#67 test models
#ziyan says run with 2*67 = 134
#run with 10 for now to debug
