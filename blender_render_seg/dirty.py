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

bpy.ops.import_scene.obj(filepath='test1.obj')
for name in bpy.data.objects.keys():
    if name not in ['Camera', 'Lamp']:
        bpy.data.objects[name].location = bpy.data.objects[name].location + mathutils.Vector((0.1, 0.1, 0.1))
        
for obj in bpy.data.objects:
    obj.select = True
    if 'mesh' in obj.name:
        obj.pass_index = 4
    obj.select = False
        
bpy.ops.import_scene.obj(filepath='test2.obj')

for obj in bpy.data.objects:
    obj.select = True
    if 'mesh' in obj.name:
        if obj.pass_index != 4:
            obj.pass_index = 3
    obj.select = False
    
for name in bpy.data.objects.keys():
    if name != 'Cube':
        bpy.data.objects[name].select = False
    else:
        bpy.data.objects[name].select = True
bpy.ops.object.delete()

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

print(dir(bpy.context.scene.render.layers["RenderLayer"]))
#exit()
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_object_index = True

rl = tree.nodes.new(type="CompositorNodeRLayers")
seg_output = tree.nodes.new(type="CompositorNodeOutputFile")
seg_mask = tree.nodes.new(type="CompositorNodeIDMask")

seg_output.base_path = '.'

print('seg_mask')
print(seg_mask.inputs.keys())
print(seg_mask.outputs.keys())
print('seg_out')
print(seg_output.inputs.keys())
print(seg_output.outputs.keys())

#exit()
#print(dir(seg_mask))
#print(dir(seg_mask.bl_description))
#exit()

if True:
    seg_mask.index = 3
    links.new(rl.outputs['IndexOB'], seg_mask.inputs['ID value'])
    links.new(seg_mask.outputs['Alpha'], seg_output.inputs['Image'])

print(rl.outputs.keys())

######
#######

#os.system('rm /tmp/idx.png0001.png')
os.system('rm idx.png')

#bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
bpy.data.scenes['Scene'].render.filepath = 'out'
seg_output.file_slots[0].path = 'idx'

bpy.ops.render.render( write_still=True )

#os.system('cp /tmp/idx.png0001.png idx.png')
#print (bpy.data.objects.keys())
#ipdb.set_trace()
