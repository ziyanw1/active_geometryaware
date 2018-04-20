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

#### GRU
## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_3' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=1 \
#     --delta=20.0 --reg_act=10 --penalty_weight=1e-4

## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_train' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=1 \
#     --delta=20.0 --reg_act=0.1 --penalty_weight=1e-4

## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_dqnVoxbn' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=5 \
#     --delta=20.0 --reg_act=1 --penalty_weight=1e-4

## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_gamma0' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0 --penalty_weight=0 --gamma=0

## GRU IG GAMMA=0 no regularization 
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_ig_longermem_gamma0' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0 --penalty_weight=0 --gamma=0

## GRU IoU GAMMA=0 no regularization
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_gamma0' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0 --penalty_weight=0 --gamma=0

## GRU IoU GAMMA=0 no regularization
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_gamma0_coef5' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=20000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.0001 --penalty_weight=0 --gamma=0 --epsilon=0.05

## GRU IoU GAMMA=0 no regularization
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_gamma0_coef5_noreg' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=20000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=16 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0 --penalty_weight=0 --gamma=0 --epsilon=0.05

## GRU IoU GAMMA=0 no regularization
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_gamma0_coef5_noreg_highR' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=20000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.1 --penalty_weight=0 --gamma=0 --epsilon=0.1 

## GRU IoU GAMMA=0 no regularization single object
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_gamma0_coef5_single' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=0 \
#     --test_every_step=50 --save_every_step=50 --max_iter=20000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=16 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.01 --penalty_weight=0 --gamma=0.9 --epsilon=0.1 --debug_single=True --finetune_dqn_only=True \
#     --pretrain_restore=True --pretrain_restore_path=log_agent/pretrain_models/best_model/model.ckpt-18400

## GRU IoU GAMMA=0 no regularization
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_gamma0.9_coef5_noreg_highR' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=20000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.01 --penalty_weight=0 --gamma=0.9 --epsilon=0.05

## GRU IoU GAMMA=0 no regularization use critic
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_gamma0_coef5_noreg_critic' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=20000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=16 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0 --penalty_weight=0 --gamma=0 --epsilon=0.05 --use_critic=True

## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_ftDQN' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=10000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.3 --penalty_weight=1e-4 --finetune_dqn=True --pretrain_restore=False

## GRU IG pretrain loss_coef=10
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_ftDQN_coef10' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=20000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.3 --penalty_weight=1e-4 --finetune_dqn=True --pretrain_restore=False

## GRU IG continue pretrain
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_ftDQN_coef10_pretrain' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=20000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.3 --penalty_weight=1e-4 --finetune_dqn=True --pretrain_restore=True \
#     --pretrain_restore_path='log_agent/pretrain_models/MVnet_coef10/pretrain_model.ckpt-20000' \
#     --burnin_start_iter=20000

## GRU IoU pretrain
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef10_gamma0' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn=True --pretrain_restore=True --gamma=0 \
#     --pretrain_restore_path='log_agent/pretrain_models/MVnet_coef10/pretrain_model.ckpt-20000'

## GRU IoU pretrain
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef5_gamma0' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=10 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.001 --penalty_weight=0 --finetune_dqn=True --pretrain_restore=True --gamma=0 \
#     --pretrain_restore_path='log_agent/pretrain_models/MVnet_coef10/pretrain_model.ckpt-20000' --epsilon=0.05

## GRU IoU pretrain
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef5_gamma0_lowR' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=10 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=0.1 \
#     --delta=20.0 --reg_act=0.001 --penalty_weight=0 --finetune_dqn=True --pretrain_restore=True --gamma=0 \
#     --pretrain_restore_path='log_agent/pretrain_models/MVnet_coef10/pretrain_model.ckpt-20000' --epsilon=0.05

## GRU IoU pretrain dqn only
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQNonly_coef10_gamma0' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn_only=True --pretrain_restore=True --gamma=0 \
#     --pretrain_restore_path='log_agent/pretrain_models/MVnet_coef10/pretrain_model.ckpt-20000'

## GRU IoU pretrain dqn only
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQNonly_coef5_gamma0' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=0 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=16 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=50 \
#     --delta=20.0 --reg_act=0.1 --penalty_weight=0 --finetune_dqn_only=True --pretrain_restore=True --gamma=0 \
#     --pretrain_restore_path='log_agent/pretrain_models/best_model/model.ckpt-18400'

## GRU IoU pretrain dqn only
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQNonly_coef5_gamma0.9' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=0 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=16 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=50 \
#     --delta=20.0 --reg_act=0.1 --penalty_weight=0 --finetune_dqn_only=True --pretrain_restore=True --gamma=0.9 \
#     --pretrain_restore_path='log_agent/pretrain_models/best_model/model.ckpt-18400'

## GRU IoU pretrain dqn only
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQNonly_coef5_gamma0_longer' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=0 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=16 \
#     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=50 \
#     --delta=20.0 --reg_act=0.1 --penalty_weight=0 --finetune_dqn_only=True --pretrain_restore=True --gamma=0 \
#     --pretrain_restore_path='log_agent/gru_nostop_iou_longermem_ftDQNonly_coef5_gamma0/log_agent/model.ckpt-10000'

## GRU IoU no regularization pretrain random explore
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_iou_longermem_ftDQN_coef10_gamma0_random_explore' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.01 --penalty_weight=0 --finetune_dqn=True --pretrain_restore=True --gamma=0 \
#     --pretrain_restore_path='log_agent/pretrain_models/MVnet_coef10/pretrain_model.ckpt-20000' --explore_mode='random'

## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_ftDQN_coef10_nolimit' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=20000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 --random_pretrain=True \
#     --use_coef=True --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.3 --penalty_weight=1e-4 --finetune_dqn=True --pretrain_restore=False --burnin_mode='nolimit'

## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_ftDQN_only' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 --burn_in_iter=20000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=1 --penalty_weight=1e-4 --finetune_dqn_only=True

## GRU IG
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_2' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.3 --penalty_weight=1e-4 --gamma=0.99

## GRU IG DQN VOX ONLY
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_longermem_voxonly' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=1 --penalty_weight=1e-4 --dqn_use_rgb=False

## GRU IG single
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_neg_recon_single' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=100 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=1 \
#     --delta=20.0 --reg_act=0 --penalty_weight=1e-4 --debug_single=True

## GRU MVNET
#python train_mvnet.py --is_training=True --task_name='gru_nostop_mvnet' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-5 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=0.01 \
#     --delta=20.0 --reg_act=10 --penalty_weight=1e-4

## GRU IoU
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_IoU_2' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=0.1 --penalty_weight=1e-5

## GRU IG PRETRAIN
#python train_active_mvnet.py --is_training=True --task_name='gru_nostop_pretrain_2' --random_pretrain=False \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=4 --burn_in_iter=10000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=10 \
#     --delta=20.0 --reg_act=1 --penalty_weight=1e-4
#### GRU

#### POOL
## POOL IG
#python train_active_mvnet.py --is_training=True --task_name='pool_nostop_neg_recon' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-5 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=0.1 \
#     --delta=20.0 --reg_act=1 --penalty_weight=1e-4

## POOL MVNET
#python train_mvnet.py --is_training=True --task_name='pool_nostop_mvnet' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-5 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=0.1 \
#     --delta=20.0 --reg_act=10 --penalty_weight=1e-4

## POOL IoU
#python train_active_mvnet.py --is_training=True --task_name='pool_nostop_iou' \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=0.01 \
#     --delta=20.0 --reg_act=1 --penalty_weight=1e-4

## POOL IG PRETRAIN
#python train_active_mvnet.py --is_training=True --task_name='pool_nostop_ig_pretrain' --random_pretrain=True \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=10000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=8 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=0.1 \
#     --delta=20.0 --reg_act=3 --penalty_weight=1e-4

## POOL IoU PRETRAIN
#python train_active_mvnet.py --is_training=True --task_name='pool_nostop_iou_pretrain' --random_pretrain=True \
#     --max_episode_length=4 --reward_type='IoU' --learning_rate=5e-4 --burn_in_length=4 --burn_in_iter=1500 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=0.1 \
#     --delta=20.0 --reg_act=10 --penalty_weight=1e-4
#### POOL

## POOL IG PRETRAIN debug
#python train_active_mvnet.py --is_training=True --task_name='overfitting' --random_pretrain=True \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 --burn_in_iter=1500 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=1 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=1 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=100 \
#     --delta=20.0 --reg_act=5 --penalty_weight=1e-4

## debug rollout go
#python train_active_mvnet.py --is_training=True --task_name='debug' --random_pretrain=False \
#     --max_episode_length=4 --reward_type='IG' --random_pretrain=True --learning_rate=5e-4 --burn_in_length=10 --burn_in_iter=10 \
#     --test_every_step=2 --save_every_step=2000 --max_iter=50000 --batch_size=2 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=1 \
#     --loss_coef=10 --if_save_eval=True --test_episode_num=2 --GBL_thread=True --reward_weight=100

### no active pool
##python train_mvnet.py --is_training=True --task_name='mvnet_pool' \
##     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
##     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
##     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='POOL' --mem_length=1000 \
##     --loss_coef=10 --if_save_eval=True --test_episode_num=20
#
### no active gru
##python train_mvnet.py --is_training=True --task_name='mvnet_gru_unseen_addstop_last' \
##     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
##     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
##     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=1000 \
##     --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True

# example for training on a single chair model
#python train_active_mvnet.py --is_training=True --task_name='active_mvnet_0322_seq4_IG_chair_single' \
#    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-4 --burn_in_length=10 \
#    --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#    --resolution=128 --category='03001627' --mem_length=1000 \
#    --train_filename_prefix='single' --val_filename_prefix='single' --test_filename_prefix='single'

## baseline 2d
#python train_active_mvnet_2d.py --is_training=True --task_name='baseline_2d' --random_pretrain=False \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-5 --burn_in_length=4 --burn_in_iter=20000 \
#     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=8 \
#     --use_coef=False --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=100 \
#     --delta=20 --reg_act=0.01

## baseline 2d
python train_active_mvnet_2d.py --is_training=True --task_name='baseline_2d_coef5' --random_pretrain=False \
     --max_episode_length=4 --reward_type='IG' --learning_rate=1e-5 --burn_in_length=4 --burn_in_iter=20000 \
     --test_every_step=200 --save_every_step=200 --max_iter=10000 --batch_size=4 --voxel_resolution=64 \
     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=8 \
     --use_coef=True --loss_coef=5 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=100 \
     --delta=20 --reg_act=0.01

## debug 2d
#python train_active_mvnet_2d.py --is_training=True --task_name='debug_2d' \
#     --max_episode_length=4 --reward_type='IG' --learning_rate=5e-4 --burn_in_length=10 \
#     --test_every_step=2 --save_every_step=2000 --max_iter=50000 --batch_size=2 --voxel_resolution=64 \
#     --resolution=128 --category='03001627' --unet_name='U_VALID' --agg_name='GRU' --mem_length=1000 \
#     --loss_coef=10 --if_save_eval=True --test_episode_num=20 --GBL_thread=True --reward_weight=100 \
#     --delta=20 --reg_act=0.01
### debug train single
#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0315_seq4_debug_single' \
#    --max_episode_length=4 --reward_type='IoU' --learning_rate=1e-5 --debug_single=True --burn_in_length=100 \
#    --test_every_step=100 --save_every_step=100

### debug train single
#CUDA_VISIBLE_DEVICES=1 python train_active_lsm.py --is_training=True --task_name='agent_0315_seq4_IG_debug_single' \
#    --max_episode_length=4 --reward_type='IG' --learning_rate=1e-5 --debug_single=True --burn_in_length=100 \
#    --test_every_step=100 --save_every_step=100
