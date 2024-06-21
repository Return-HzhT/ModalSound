import sys
sys.path.append("..")
from src.model import ModalSoundObj, Material, MatSet, get_spherical_surface_points
from src.visualize import CombinedFig
import numpy as np
from tqdm import tqdm
import os

file = "guitar_0250"
voxel_file_name = "./dataset/voxel/"+file+".npz"
eigen_file_name = "./dataset/eigen/"+file+".npz"
ffat_file_name = "./dataset/ffat/"+file+".npy"
# pcd_file_name = "./dataset/pcd_dataset/"+file+".npz"
pcd_file_name = "./dataset/final_pcd_dataset/"+file+".npz"
voxel_data = np.load(voxel_file_name)
eigen_data = np.load(eigen_file_name)
ffat_data = np.load(ffat_file_name)
pcd_data = np.load(pcd_file_name)

surf_vertices = voxel_data['surf_vertices']
surf_triangles = voxel_data['surf_triangles']
vertices = voxel_data['vertices']
tets = voxel_data['tets']
bbox_min = voxel_data['bbox_min']
bbox_max = voxel_data['bbox_max']
surf_normals = voxel_data['surf_normals']
size = voxel_data['size']
center = voxel_data['center']

pcd = pcd_data['pcd']
pcd_eigenvectors = pcd_data['eigenvectors']

# print(surf_vertices.shape,surf_triangles.shape)
# print(pcd.shape)

obj = ModalSoundObj(surf_vertices,
                    surf_triangles,
                    vertices,
                    tets,
                    bbox_min,
                    bbox_max,
                    surf_normals,
                    size,
                    center)

# FFAT_map_points = get_spherical_surface_points(obj.surf_vertices)
# FFAT_map = np.abs(ffat_data[0])

# print(FFAT_map_points.shape, FFAT_map.shape)

# modes = (eigen_data['modes']**2).sum(axis=1)

# print(modes.shape)
# print(modes[:,0].shape)

# print(eigen_data['modes'].shape)
# # print(modes.shape)

# print(FFAT_map_points.shape,FFAT_map.shape)

# visualize the first mode and its FFAT map
# CombinedFig().add_mesh(obj.surf_vertices, obj.surf_triangles, modes[:, 0]).add_points(
#     FFAT_map_points, FFAT_map
# ).show()

# CombinedFig().add_mesh(obj.surf_vertices, obj.surf_triangles, modes[:, 0]).add_points(
#     FFAT_map_points, FFAT_map
# ).save("./tmp.png")

# CombinedFig().add_mesh(obj.surf_vertices, obj.surf_triangles).show()

print(pcd.shape)
print(pcd_eigenvectors.shape)

vecs = (pcd_eigenvectors**2).sum(axis=1)
print(vecs.shape)
CombinedFig().add_points(pcd, vecs[:,1]).show()

# print(pcd_eigenvectors.shape)