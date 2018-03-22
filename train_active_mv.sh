#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0305_seq5' 

#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0305_seq4' \
#    --max_episode_length=4 

#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0311_seq4_IG' \
#    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-3

CUDA_VISIBLE_DEVICES=1 python train_active_mvnet.py --is_training=True --task_name='active_mvnet_0322_seq4_IG_chair' \
    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 \
    --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
    --resolution=128 --category='03001627', --mem_length=1000

### debug train single
#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0315_seq4_debug_single' \
#    --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-5 --debug_single=True --burn_in_length=100 \
#    --test_every_step=100 --save_every_step=100

### debug train single
#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0315_seq4_IG_debug_single' \
#    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-5 --debug_single=True --burn_in_length=100 \
#    --test_every_step=100 --save_every_step=100
