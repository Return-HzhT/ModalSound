import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import os
import random

class FfatDataset(Dataset):
    def __init__(self, phase):
        self.dataset_dir = "../split_dataset/" + phase
        self.file_list = os.listdir(self.dataset_dir)
        print(phase, 'set size: ', len(self.file_list))
        self._indices = None
        self.transform = None

    def len(self):
        return len(self.file_list)
    

    def get(self, index):
        data_name = self.file_list[index]
        data_path = os.path.join(self.dataset_dir, data_name)
        data = np.load(data_path)
        pcd = data['pcd']
        vals = data['eigenvalues']
        vecs = data['eigenvectors']
        ffat = data['ffat']
        
        return Data(pos = torch.FloatTensor(pcd), vals = torch.FloatTensor(vals), vecs = torch.FloatTensor(vecs), ffat = torch.FloatTensor(ffat))
    
    def index_select(self,idx):
        return self.get(idx)


