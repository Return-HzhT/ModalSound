import sys
sys.path.append("..")
from src.model import ModalSoundObj, Material, MatSet, get_spherical_surface_points
from src.visualize import CombinedFig
import numpy as np
from tqdm import tqdm
import os

file_list = "../dataset/voxel/"
out_list = "../dataset/"
connect_data = np.load("../dataset/voxel_connect.npy", allow_pickle=True)
dict = connect_data.item()

def single_pipeline(voxel_file_name):
    data = np.load(voxel_file_name)

    surf_vertices = data['surf_vertices']
    surf_triangles = data['surf_triangles']
    vertices = data['vertices']
    tets = data['tets']
    bbox_min = data['bbox_min']
    bbox_max = data['bbox_max']
    surf_normals = data['surf_normals']
    size = data['size']
    center = data['center']

    obj = ModalSoundObj(surf_vertices,
                        surf_triangles,
                        vertices,
                        tets,
                        bbox_min,
                        bbox_max,
                        surf_normals,
                        size,
                        center)
    
    if obj.surf_triangles.shape[0]>10000:
        return
    
    obj.modal_analysis(32, Material(MatSet.Plastic))

    FFAT_map_points = get_spherical_surface_points(obj.surf_vertices)
    total_ffat_map = np.zeros((32, FFAT_map_points.shape[0]))
    residual_list = []

    for mode_id in range(32):
        ffat, residual = obj.solve_ffat_map(mode_id, FFAT_map_points)
        FFAT_map = np.abs(ffat)
        total_ffat_map[mode_id] = FFAT_map

        residual_list.append(residual)

    residual_numpy = np.array([np.array(res) for res in residual_list], dtype=object)
    
    eigen_out_path = os.path.join(out_list, "eigen", os.path.basename(filename).replace('.npz', '.npz'))
    FFAT_map_out_path = os.path.join(out_list, "ffat", os.path.basename(filename).replace('.npz', '.npy'))
    residual_out_path = os.path.join(out_list, "residual", os.path.basename(filename).replace('.npz', '.npy'))

    np.savez(eigen_out_path, eigenvalues = obj.eigenvalues, modes = obj.modes)
    np.save(FFAT_map_out_path, total_ffat_map)
    np.save(residual_out_path, residual_numpy)

sorted_file_list = os.listdir(file_list)
sorted_file_list.sort()


for filename in tqdm(sorted_file_list):
    voxel_file_name = os.path.join(file_list,filename)
    print(voxel_file_name)
    now_file_name = filename.replace('.npz', '')
    if dict[now_file_name] == 0:
        print("Continue.")
        continue
    try:
        single_pipeline(voxel_file_name)
    except:
        continue