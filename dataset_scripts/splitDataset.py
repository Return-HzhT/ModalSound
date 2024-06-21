from tqdm import tqdm
import os
import random
import numpy as np

out_dir = "../split_dataset/"
if os.path.exists(out_dir):
    os.system("rm -r %s" % out_dir)
os.mkdir(out_dir)
os.mkdir(out_dir + "train/")
os.mkdir(out_dir + "valid/")
os.mkdir(out_dir + "test/")

def process_dataset(phase):
    pcd_sample = 4
    pcd_dataset_dir = "../dataset/final_pcd_dataset"
    
    out_dataset_dir = "../split_dataset/"+phase+"/"

    now_data_list = "../"+phase+".txt"

    file_list = None
    with open(now_data_list, 'r') as f:
        file_list = f.readlines()
    
    for data in tqdm(file_list):
        for i in range(pcd_sample):
            pcd_dataset_path = os.path.join(pcd_dataset_dir,data[:-1]+"_"+str(i)+".npz")
            out_data_path = os.path.join(out_dataset_dir,data[:-1]+"_"+str(i)+".npz")
        
            pcd_dataset = np.load(pcd_dataset_path)
            pcd = pcd_dataset['pcd']
            pcd_eigenvalues = pcd_dataset['eigenvalues']
            pcd_eigenvectors = pcd_dataset['eigenvectors']
            ffat = pcd_dataset['ffat']
            eigenvalues_mask = pcd_dataset['eigenvalues_mask']
            masked_eigenvectors = pcd_dataset['masked_eigenvectors']
            np.savez(out_data_path, pcd = pcd, eigenvalues = pcd_eigenvalues, eigenvectors = pcd_eigenvectors, ffat = ffat, eigenvalues_mask = eigenvalues_mask, masked_eigenvectors = masked_eigenvectors)
    return

if __name__ == '__main__':
    root_dir = "../"
    file_list = np.load('../dataset/residual_ok.npy').tolist()

    random.shuffle(file_list)
    length = len(file_list)
    idx = [0, length * 0.1, length * 0.2, length]
    phase = ['test', 'valid', 'train']
    for i in range(len(phase)):
        with open(os.path.join(root_dir, f'{phase[i]}.txt'),'w') as f:
            lst = file_list[int(idx[i]):int(idx[i+1])]
            for line in lst:
                f.write(os.path.basename(line).replace('.npy','') + '\n')
    
    process_dataset("train")
    process_dataset("valid")
    process_dataset("test")