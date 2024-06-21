import sys
sys.path.append('..')
from external.sample import ImportanceSampler
import numpy as np
import os
import torch
from eigenvalue_model.point_transformer import Net as valueNet
from eigenvector_model.pointnet2 import Net as vectorNet
from ffat_model.point_transformer import Net as ffatNet
from torch_geometric.data import Data
import open3d as o3d

def read_obj_with_open3d(filename):
    # 读取OBJ文件
    mesh = o3d.io.read_triangle_mesh(filename)
    
    if mesh.is_empty():
        print("The mesh is empty!")
        return None, None
    
    surf_vertices = np.asarray(mesh.vertices)
    surf_triangles = np.asarray(mesh.triangles)
    
    return surf_vertices, surf_triangles

def compute_sample_r(vertices, triangles, n):
    vertices = vertices.float()
    triangles = triangles.long()
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    cross_product = torch.cross(edge1, edge2, dim=1)
    triangle_areas = 0.5 * torch.norm(cross_product, dim=1)
    total_area = torch.sum(triangle_areas)
    return (total_area.item() / (2 * n)) ** 0.5

filename = "./dataset/mesh/bowl_0001.obj"

# pcd_dataset_folder_path = "./dataset/pcd_dataset"
# data_path = os.path.join(pcd_dataset_folder_path, os.path.basename(filename).replace('.npy', '_0.npz'))
# data = np.load(data_path)
# pcd = data['pcd']

surf_vertices, surf_triangles = read_obj_with_open3d(filename)
scale = 0.2
if surf_vertices is not None and surf_triangles is not None:
    bbox_min = surf_vertices.min(axis=0)
    bbox_max = surf_vertices.max(axis=0)
    surf_vertices = (surf_vertices - (bbox_max + bbox_min) / 2) / (bbox_max - bbox_min).max()
    surf_vertices = surf_vertices * scale

pcd = None
sample_num = 1000
if surf_vertices is not None and surf_triangles is not None:
    vertices = torch.from_numpy(surf_vertices).float().cuda()
    triangles = torch.from_numpy(surf_triangles).int().cuda()
    importance = torch.ones(len(triangles)).float().cuda()
    r = compute_sample_r(vertices, triangles, sample_num)
    sampler = ImportanceSampler(vertices, triangles, importance, 50000)
    sampler.update()
    sampler.poisson_disk_resample(r, 4)
    pcd = sampler.points.cpu().numpy()

value_model = valueNet().cuda()
value_model.load_state_dict(torch.load("./eigenvalue_model/train/best.pth"))

vector_model = vectorNet().cuda()
vector_model.load_state_dict(torch.load("./eigenvector_model/train/best.pth"))

ffat_model = ffatNet().cuda()
ffat_model.load_state_dict(torch.load("./ffat_model/train/best.pth"))

data = Data(pos=torch.from_numpy(pcd))
data.batch = torch.zeros((pcd.shape[0]),dtype=torch.int64)

eigenvalues_mask = value_model(data.cuda())
eigenvalues_mask  = eigenvalues_mask.cpu().detach().numpy()
eigenvalues_mask  = eigenvalues_mask.reshape(-1)


def fre2val(fre):
    return (2 * np.pi * fre)**2

std_data = np.load("./dataset/total_std.npz")
eigenvalue_std = std_data['eigenvalue_std']
eigenvector_std = std_data['eigenvector_std']
ffat_std = std_data['ffat_std']

f_min, f_max = 20, 20000
f_min = fre2val(f_min)
f_max = fre2val(f_max)
f_min /= eigenvalue_std
f_max /= eigenvalue_std
f_min = np.log((f_min+1e-3)/1e-3)
f_max = np.log((f_max+1e-3)/1e-3)

min_value, max_value = f_min, f_max

delta = (max_value - min_value) / 32
eigen_mask_list = [(min_value + i*delta) for i in range(33)]

tmp_eigen = []
for i in range(32):
    if eigenvalues_mask[i] > 0.5:
        tmp_eigen.append((eigen_mask_list[i]+eigen_mask_list[i+1])/2)

tmp_eigen = np.array(tmp_eigen, dtype=np.float32)
eigenvalues = np.exp(tmp_eigen) * 1e-3 - 1e-3
eigenvalues = eigenvalues * eigenvalue_std

eigenvectors = vector_model(data.cuda())
eigenvectors = eigenvectors.cpu().detach().numpy()
eigenvectors = eigenvectors[:,:,eigenvalues_mask>0.5]
eigenvectors = np.exp(eigenvectors) * 1e-3 - 1e-3
eigenvectors = eigenvectors * eigenvector_std

data = Data(pos=torch.from_numpy(pcd), vals=torch.from_numpy(tmp_eigen))
ffat = ffat_model(data.cuda())
ffat = ffat.cpu().detach().numpy()

ffat = np.exp(ffat) * 1e-3 - 1e-3
ffat = ffat * ffat_std

np.savez("./bowl.npz", pcd=pcd, eigenvalues=eigenvalues, eigenvectors=eigenvectors, ffat=ffat)

