import sys
sys.path.append("..")
from src.model import ModalSoundObj, Material, MatSet, get_spherical_surface_points
from src.visualize import CombinedFig
import numpy as np
from tqdm import tqdm
import os
import networkx as nx


file_list = "../dataset/voxel"
connect_dict = {}

for file in tqdm(os.listdir(file_list)):
    data = np.load(os.path.join(file_list, file))
    n = data['surf_vertices'].shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(n))

    triangles = data['surf_triangles']
    edge = []
    for i in range(triangles.shape[0]):
        edge.append((triangles[i][0],triangles[i][1]))
        edge.append((triangles[i][1],triangles[i][2]))
        edge.append((triangles[i][2],triangles[i][0]))

    G.add_edges_from(edge)

    file_name = file.replace(".npz", "")
    if nx.is_connected(G):
        connect_dict[file_name] = 1
    else:
        connect_dict[file_name] = 0

np.save("../dataset/voxel_connect.npy", connect_dict)

data = np.load("../dataset/voxel_connect.npy", allow_pickle=True)
print(data)