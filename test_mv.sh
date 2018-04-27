
## test random
python test_active_mvnet.py --LOG_DIR=log_agent/ --task_name=gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn --test_iter=20000 \
    --test_episode_num=1317 --pretrain_restore=True --if_bn=False \
    --pretrain_restore_path=log_agent/gru_nostop_iou_longermem_ftDQN_coef1_lastrecon_NObn/log_agent/pretrain_model.ckpt-20000 \
    --test_random=False --category='03001627' \
## set test random to True to use random policy
