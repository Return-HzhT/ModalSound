import sys
sys.path.append('..')
import numpy as np
import os
from tqdm import tqdm
from external.sample import ImportanceSampler
import torch

data_list = np.load('../dataset/residual_ok.npy')
voxel_folder_path = "../dataset/voxel"
eigen_folder_path = "../dataset/eigen"
ffat_folder_path = "../dataset/ffat"
pcd_dataset_folder_path = "../dataset/pcd_dataset"

total_eigenvalues_square_sum = 0.0
total_eigenvectors_square_sum = 0.0
total_ffat_square_sum = 0.0

total_eigenvalues_cnt = 0
total_eigenvectors_cnt = 0
total_ffat_cnt = 0

def compute_sample_r(vertices, triangles, n):
    vertices = vertices.float()
    triangles = triangles.long()
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    cross_product = torch.cross(edge1, edge2, dim=1)
    triangle_areas = 0.5 * torch.norm(cross_product, dim=1)
    total_area = torch.sum(triangle_areas)
    return (total_area.item() / (2 * n)) ** 0.5

sample_num = 1000
sample_cnt = 4

for filename in tqdm(data_list):
    for i in range(sample_cnt):
        voxel_path = os.path.join(voxel_folder_path, os.path.basename(filename).replace('.npy', '.npz'))
        eigen_path = os.path.join(eigen_folder_path, os.path.basename(filename).replace('.npy', '.npz'))
        ffat_path = os.path.join(ffat_folder_path, os.path.basename(filename).replace('.npy', '.npy'))
        voxel_data = np.load(voxel_path)
        eigen_data = np.load(eigen_path)
        ffat_data = np.load(ffat_path)
    
        vertices = torch.from_numpy(voxel_data['surf_vertices']).float().cuda()
        triangles = torch.from_numpy(voxel_data['surf_triangles']).int().cuda()
        importance = torch.ones(len(triangles)).float().cuda()
        r = compute_sample_r(vertices, triangles, sample_num)
        sampler = ImportanceSampler(vertices, triangles, importance, 50000)
        sampler.update()
        sampler.poisson_disk_resample(r, 4)

        eigen_vals, eigen_vecs = eigen_data['eigenvalues'], eigen_data['modes']
        uniform_pc = sampler.points.cpu().numpy()

        eigen_vecs = torch.from_numpy(eigen_vecs).float().cuda()
        triangle_eigenvecs = (eigen_vecs[triangles[:, 0]] + eigen_vecs[triangles[:, 1]] + eigen_vecs[triangles[:, 2]]) / 3
        points_eigenvecs = sampler.get_points_eigenvector(triangle_eigenvecs)

        pcd_dataset_out_path = os.path.join(pcd_dataset_folder_path, os.path.basename(filename).replace('.npy', '')+'_'+str(i)+'.npz')

        # 将特征值及对应的特征向量和ffat按特征值升序排列
        idx = np.argsort(eigen_vals)
        eigen_vals = eigen_vals[idx]
        eigen_vecs = points_eigenvecs[:,:,idx].cpu().numpy()
        ffat = ffat_data[idx,:]
        
        np.savez(pcd_dataset_out_path, pcd = uniform_pc, eigenvalues = eigen_vals, eigenvectors = eigen_vecs, ffat = ffat)

        total_eigenvalues_square_sum += np.sum(eigen_vals ** 2)
        total_eigenvectors_square_sum += np.sum(eigen_vecs ** 2)
        total_ffat_square_sum += np.sum(ffat ** 2)
        total_eigenvalues_cnt += eigen_vals.size
        total_eigenvectors_cnt += eigen_vecs.size
        total_ffat_cnt += ffat.size

total_eigenval_std = np.sqrt(total_eigenvalues_square_sum / total_eigenvalues_cnt)
total_eigenvec_std = np.sqrt(total_eigenvectors_square_sum / total_eigenvectors_cnt)
total_ffat_std = np.sqrt(total_ffat_square_sum / total_ffat_cnt)

pcd_dataset_folder_path = "../dataset/pcd_dataset"
scaled_out_folder_path = "../dataset/scaled_pcd_dataset"

np.savez("../dataset/total_std.npz",eigenvalue_std = total_eigenval_std, eigenvector_std = total_eigenvec_std, ffat_std = total_ffat_std)

for filename in tqdm(os.listdir(pcd_dataset_folder_path)):
    data_path = os.path.join(pcd_dataset_folder_path, filename)
    scaled_out_path = os.path.join(scaled_out_folder_path, filename)

    data = np.load(data_path)
    pcd, eigenvalues, eigenvectors, ffat = data['pcd'], data['eigenvalues'], data['eigenvectors'], data['ffat']

    # 除以总体的标准差
    eigenvalues = eigenvalues / total_eigenval_std
    eigenvectors = np.abs(eigenvectors / total_eigenvec_std)
    ffat = ffat / total_ffat_std

    eigenvalues = np.log((eigenvalues + 1e-3) / 1e-3)
    eigenvectors = np.log((eigenvectors + 1e-3) / 1e-3)
    ffat = np.log((ffat + 1e-3) / 1e-3)

    np.savez(scaled_out_path, pcd=pcd, eigenvalues=eigenvalues, eigenvectors=eigenvectors, ffat=ffat)