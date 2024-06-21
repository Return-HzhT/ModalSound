import os
import numpy as np

file_list = "../dataset/residual"
ok_list = []

for filename in os.listdir(file_list):
    data_path = os.path.join(file_list, filename)
    data = np.load(data_path, allow_pickle=True)
    flag = True
    for i in data:
        if len(i)<=1:
            flag = False
            break
        if np.any(np.isnan(i.astype(np.float64))):
            flag = False
            break 
    if flag:
        ok_list.append(filename)
np.save("../dataset/residual_ok.npy", ok_list)