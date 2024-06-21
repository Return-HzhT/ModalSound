import sys
sys.path.append("..")
from src.model import ModalSoundObj, Material, MatSet, get_spherical_surface_points
from src.visualize import CombinedFig
import numpy as np
from tqdm import tqdm
import os

file_list = "../dataset/mesh/"
out_list = "../dataset/"

def single_pipeline(obj_file_name):
    voxel_out_path = os.path.join(out_list, "voxel", os.path.basename(filename).replace('.obj', '.npz'))
    if (os.path.exists(voxel_out_path)):
        return
    
    obj = ModalSoundObj(obj_file_name)
    if obj.timeout_flag:
        return
    obj.normalize(0.2)
    
    np.savez(voxel_out_path,
            surf_vertices = obj.surf_vertices,
            surf_triangles = obj.surf_triangles,
            vertices = obj.vertices,
            tets = obj.tets,
            bbox_min = obj.bbox_min,
            bbox_max = obj.bbox_max,
            surf_normals = obj.surf_normals,
            size = obj.size,
            center = obj.center)

sorted_file_list = os.listdir(file_list)
sorted_file_list.sort()

last_file_name = "airplane_0133.obj"
continue_flag = True

for filename in tqdm(sorted_file_list):
    if not filename == last_file_name and continue_flag == True:
        print("Continue.")
        continue
    elif filename == last_file_name:
        continue_flag = False
    obj_file_name = os.path.join(file_list,filename)
    print(obj_file_name)

    try:
        single_pipeline(obj_file_name)
    except:
        continue