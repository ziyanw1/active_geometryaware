import bpy
import numpy as np
from mathutils import Vector
from stl import mesh
import stl

#pth = "/home/ricson/Downloads/BAC_Batman70s_rocksteady/batman.obj"
#out = "/home/ricson/Downloads/BAC_Batman70s_rocksteady/batman_out.obj"

pth = "../mugs/1038e4eac0e18dcce02ae6d2a21d494a/model.obj"
out = "../new_RL3/hindsight_experience_replay/HER/envs/mjc/meshes/stls/model.stl"

#first clear everything
bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.import_scene.obj(filepath=pth)

############ 
#now we need to get the maximum and minimum of all vertices

#not sure what this does???
bpy.ops.object.transform_apply( rotation = True, scale = True)

vtx2npy = lambda v: np.array(v.co)

vertices = []
for obj in bpy.context.scene.objects: 
    if obj.type == 'MESH':
        vertices.extend(map(vtx2npy, obj.data.vertices))

vertices = np.array(vertices) #Nx3

mins = np.min(vertices, axis = 0)
maxs = np.max(vertices, axis = 0)
centers = (maxs+mins)/2.0
size = np.max(maxs-mins)



vertices = []
vertex_count = 0
for obj in bpy.context.scene.objects:

    vertex_count += len(obj.data.vertices)
    
    if obj.type == 'MESH':
        for v in obj.data.vertices:
            
            #this normalizes it into the (-1,1) 3-cube
            v.co -= Vector(centers)
            v.co /= size/2.0 #so largest dim is 2.0

print("there are", vertex_count, "vertices")

#############
#exporting

#bpy.ops.export_scene.obj(filepath=out)
bpy.ops.export_mesh.stl(filepath=out)

def invert(msh):
    for face, _ in enumerate(msh.vectors):
        msh.vectors[face] = msh.vectors[face][::-1]
    return msh

def concat(msh1, msh2):
    return mesh.Mesh(np.concatenate([msh1.data, msh2.data]))

#sometimes the mesh is bad / inverted, so we make combine the mesh with an inverted copy

msh = mesh.Mesh.from_file(out)
msh_ = mesh.Mesh.from_file(out)

msh = concat(msh, invert(msh_))

msh.save(out, mode = stl.Mode.BINARY)

