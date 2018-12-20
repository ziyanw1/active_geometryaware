# MVnet active

## Data preparation

All rendering datas are located in data/data_cache/blender_renderings/$CATEGORY_INDEX/res$RESOLUTION_$NAME/$OBJECT_NAME

The files in each directory contains RGB and inverse depth data. The nameing rules are RGB_$AZIMUTH_$ELEVATION.jpg invZ_$AZIMUTH_$ELEVATION.npy

To generate data of single object scene, you can run the following command.

`blender blank.blend -b -P data/render_scripts/render_depth_pair_lambert_main.py -- $CATE_NUM $LIST_PATH $DATASET_NAME $RESOLUTION`

To generate data of double object scene, you can run the following command.

`blender blank.blend -b -P render_depth_pair_lambert_main_double.py -- $LIST1_PATH $LIST2_PATH $DATASET_NAME $RESOLUTION`

## Train MVnet active

To train active mvnet, run the following command

`python train_active_mvnet.py --task_name=$TASK_NAME --learning_rate=1e-4 --max_iter=10000 --batch_size=4 --voxel_resolution=64 --resolution=128
`

## Test MVnet active

To test active mvnet, run the following command

`python test_active_mvnet.py --LOG_DIR=$LOG_DIR --task_name=$TASK_NAME --test_iter=$ITER_OF_RESTORE_MODEL --test_episode_num=$NUM_TEST_EPISODES --pretrain_restore=True --pretrain_restore_path=$PATH_TO_RESTORE_MODEL 
`

## Info about branches

The `seg2` branch should be used to run experiments involving object segmentation. The `master` branch is appropriate for reconstruction tasks.

