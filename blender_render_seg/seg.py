#!/usr/bin/python3
# -*- coding: utf-8 -*-

#import ipdb
import os
import bpy
import sys
import math
import random
import numpy as np
import mathutils
from scipy.misc import imread, imsave

idx1 = 234
idx2 = 345

bpy.ops.import_scene.obj(filepath='test1.obj')
for name in bpy.data.objects.keys():
    if name not in ['Camera', 'Lamp']:
        bpy.data.objects[name].location = bpy.data.objects[name].location + mathutils.Vector((0.1, 0.1, 0.1))
        
for obj in bpy.data.objects:
    obj.select = True
    if 'mesh' in obj.name:
        obj.pass_index = idx1
    obj.select = False
        
bpy.ops.import_scene.obj(filepath='test2.obj')

for obj in bpy.data.objects:
    obj.select = True
    if 'mesh' in obj.name:
        if obj.pass_index != idx1:
            obj.pass_index = idx2
    obj.select = False

for name in bpy.data.objects.keys():
    if name != 'Cube':
        bpy.data.objects[name].select = False
    else:
        bpy.data.objects[name].select = True
bpy.ops.object.delete()

####

bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
bpy.data.objects['Lamp'].data.energy = 1.0
camObj = bpy.data.objects['Camera']

camObj.data.lens_unit = 'FOV'
camObj.data.angle = 0.2

bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = 0.5
bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

####

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
for n in tree.nodes:
    tree.nodes.remove(n)

bpy.context.scene.render.layers["RenderLayer"].use_pass_object_index = True

rl = tree.nodes.new(type="CompositorNodeRLayers")
seg_out_1 = tree.nodes.new(type="CompositorNodeOutputFile")
seg_mask_1 = tree.nodes.new(type="CompositorNodeIDMask")
seg_out_2 = tree.nodes.new(type="CompositorNodeOutputFile")
seg_mask_2 = tree.nodes.new(type="CompositorNodeIDMask")

seg_out_1.base_path = '.'
seg_out_2.base_path = '.'

seg_mask_1.index = idx1
seg_mask_2.index = idx2

links.new(rl.outputs['IndexOB'], seg_mask_1.inputs['ID value'])
links.new(seg_mask_1.outputs['Alpha'], seg_out_1.inputs['Image'])
links.new(rl.outputs['IndexOB'], seg_mask_2.inputs['ID value'])
links.new(seg_mask_2.outputs['Alpha'], seg_out_2.inputs['Image'])

######

bpy.data.scenes['Scene'].render.filepath = 'out'
seg_out_1.file_slots[0].path = 'idx1'
seg_out_2.file_slots[0].path = 'idx2'
bpy.ops.render.render( write_still=True )

mask1 = imread('idx10001.png')
mask1 = mask1[:,:,0] * mask1[:,:,3]

mask2 = imread('idx20001.png')
mask2 = mask2[:,:,0] * mask2[:,:,3]

bg = ((np.ones_like(mask1)-mask1-mask2) > 0).astype(mask1.dtype)

imsave(
    'out.png',
    np.stack([mask1, mask2, bg], axis = 2)*255.0
)
