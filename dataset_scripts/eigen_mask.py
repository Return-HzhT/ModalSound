import os
import numpy as np
from tqdm import tqdm

pcd_dataset_folder_path = "../dataset/scaled_pcd_dataset"
out_dataset_folder_path = "../dataset/final_pcd_dataset"

def fre2val(fre):
    return (2 * np.pi * fre)**2

std_data = np.load("../dataset/total_std.npz")
eigenvalue_std = std_data['eigenvalue_std']

f_min, f_max = 20, 20000
f_min = fre2val(f_min)
f_max = fre2val(f_max)
f_min /= eigenvalue_std
f_max /= eigenvalue_std
f_min = np.log((f_min+1e-3)/1e-3)
f_max = np.log((f_max+1e-3)/1e-3)

min_value, max_value = f_min, f_max
print(min_value, max_value)

delta = (max_value - min_value) / 32
eigen_mask_list = [(min_value + i*delta) for i in range(33)]

for filename in tqdm(os.listdir(pcd_dataset_folder_path)):
    data_path = os.path.join(pcd_dataset_folder_path, filename)
    data = np.load(data_path)
    pcd, eigenvalues, eigenvectors, ffat = data['pcd'], data['eigenvalues'], data['eigenvectors'], data['ffat']

    # 处理特征值掩码
    eigenvalues_mask = np.zeros_like(eigenvalues)
    final_eigenvectors = np.zeros_like(eigenvectors)

    # 分别考察32个区间
    for i in range(32):
        # 计算特征值掩码
        flag = False
        mask_begin, mask_end = eigen_mask_list[i], eigen_mask_list[i+1]
        for j in range(32):
            now_value = eigenvalues[j]
            if mask_begin <= now_value and now_value <= mask_end:
                flag = True
                break
        if flag == True:
            eigenvalues_mask[i] = 1

        if eigenvalues_mask[i] == 0:
             continue
        
        # 合并特征向量
        now_cnt = 0
        for j in range(32):
            now_value = eigenvalues[j]
            if mask_begin <= now_value and now_value <= mask_end:
                final_eigenvectors[:,:,i] += eigenvectors[:,:,j]
                now_cnt += 1
        final_eigenvectors[:,:,i] /= now_cnt

    out_data_path = os.path.join(out_dataset_folder_path, filename)
    np.savez(out_data_path, pcd = pcd, eigenvalues = eigenvalues, eigenvectors = eigenvectors, eigenvalues_mask = eigenvalues_mask, masked_eigenvectors = final_eigenvectors, ffat = ffat)