#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0305_seq5' 

#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0305_seq4' \
#    --max_episode_length=4 

#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0311_seq4_IG' \
#    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-3

## debug
#python train_active_mvnet.py --is_training=True --task_name='debug' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
#     --test_every_step=2000 --save_every_step=2000 --max_iter=50000 --batch_size=2 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=1000 \
#     --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=100

## max pooling
#python train_active_mvnet.py --is_training=True --task_name='asdf2' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=1000 \
#     --loss_coef=10 --if_save_eval=True --test_episode_num=20

## GRU
#python train_active_mvnet.py --is_training=True --task_name='gru_0331' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
#     --test_every_step=2000 --save_every_step=2000 --max_iter=50000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=1000 \
#     --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=100

## no active pool
#python train_mvnet.py --is_training=True --task_name='mvnet_pool' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=1000 \
#     --loss_coef=10 --if_save_eval=True --test_episode_num=20

## no active gru
#python train_mvnet.py --is_training=True --task_name='mvnet_gru_0331' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
#     --test_every_step=2000 --save_every_step=2000 --max_iter=50000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=1000 \
#     --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True

# example for training on a single chair model
#python train_active_mvnet.py --is_training=True --task_name='active_mvnet_0322_seq4_IG_chair_single' \
#    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 \
#    --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#    --resolution=128 --category='03001627' --mem_length=1000 \
#    --train_filename_prefix='single' --val_filename_prefix='single' --test_filename_prefix='single'

## debug 2d
python train_active_mvnet_2d.py --is_training=True --task_name='debug_2d' \
     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
     --test_every_step=2000 --save_every_step=2000 --max_iter=50000 --batch_size=2 --voxel_resolution=64 \
     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=1000 \
     --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=100
### debug train single
#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0315_seq4_debug_single' \
#    --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-5 --debug_single=True --burn_in_length=100 \
#    --test_every_step=100 --save_every_step=100

### debug train single
#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0315_seq4_IG_debug_single' \
#    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-5 --debug_single=True --burn_in_length=100 \
#    --test_every_step=100 --save_every_step=100
