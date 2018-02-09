# MVnet active

## train script:

'python train_ae_rgb2d_persp_lmdb.py'

args are defined in variable FLAGS. 
## network models and lmdb loader

autoencoder (RGB -> inverse depth) are defined in models/ae_rgb2depth.py.

lmdb loader is defined in models/rgb2_lmdb_loader.py

## lmdb generator

the script for generating lmdb data is data/write_lmdb_rgbAndDepth.py

## data

All rendering datas are located in data/data_cache/blender_renderings/$CATEGORY_INDEX/res$RESOLUTION_$NAME/$OBJECT_NAME

The files in each directory contains RGB and inverse depth data. The nameing rules are RGB_$AZIMUTH_$ELEVATION.jpg invZ_$AZIMUTH_$ELEVATION.npy

