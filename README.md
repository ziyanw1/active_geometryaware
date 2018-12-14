# MVnet active

## lmdb generator

the script for generating lmdb data is data/write_lmdb_rgbAndDepth.py

## data

All rendering datas are located in data/data_cache/blender_renderings/$CATEGORY_INDEX/res$RESOLUTION_$NAME/$OBJECT_NAME

The files in each directory contains RGB and inverse depth data. The nameing rules are RGB_$AZIMUTH_$ELEVATION.jpg invZ_$AZIMUTH_$ELEVATION.npy

## train script

To train rgb to depth network, run the following command

`python train_ae_rgb2d_persp_lmdb.py`

To train active mvnet, run the following command

`python train_active_mvnet.py --task_name=$TASK_NAME --learning_rate=1e-4 \
     --max_iter=10000 --batch_size=4 --voxel_resolution=64 --resolution=128
`

## test script

To test rgb to depth network, run

To test active mvnet, run the following command

`python test_active_mvnet.py --LOG_DIR=$LOG_DIR --task_name=$TASK_NAME --test_iter=$ITER_OF_RESTORE_MODEL \
    --test_episode_num=$NUM_OF_TOTAL_TEST_EPISODES --pretrain_restore=True --pretrain_restore_path=$PATH_TO_RESTORE_MODEL 
`
