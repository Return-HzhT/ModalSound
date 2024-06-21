import sys
sys.path.append("..")
from dataset import EigenvectorDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from src.visualize import CombinedFig
import os
from pointnet2 import Net
import time
import numpy as np
import cv2
from tqdm import tqdm

torch.cuda.set_device(4)

log_dir = "./evaluation/"
if os.path.exists(log_dir):
    os.system("rm -r %s" % log_dir)
writer = SummaryWriter(log_dir)

test_dataset = EigenvectorDataset('test')

batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Net().cuda()
model.load_state_dict(torch.load("./train/best.pth",map_location={'cuda:0': 'cuda:4'}))

vec_loss_fn = torch.nn.MSELoss()

@torch.no_grad()
def test():
    model.eval()

    total_loss = 0
    total_smape = 0
    now_cnt = 0

    for data in tqdm(test_loader):
        data = data.cuda()
        vec_pred = model(data)

        mask = data.mask.bool().reshape(batch_size, 32)
        vec_pred = vec_pred.permute(1, 2, 0).reshape(3,-1).permute(1, 0)
        vec_gt = data.vecs.permute(1, 2, 0).reshape(3,-1).permute(1, 0)
        points_mask = mask[data.batch].permute(1, 0).reshape(-1)

        vec_pred = vec_pred[points_mask,:]
        vec_gt = vec_gt[points_mask,:]

        loss = vec_loss_fn(vec_pred, vec_gt)

        total_smape += torch.mean(torch.abs(torch.abs(vec_pred) - torch.abs(vec_gt)) / ((torch.abs(vec_pred) + torch.abs(vec_gt)) / 2))
        total_loss += float(loss)

        now_cnt += 1

    total_loss /= len(test_loader)
    total_smape /= len(test_loader)
    return total_loss, total_smape


obj_sum = len(test_loader.dataset)
start_time = time.time()
test_loss, mean_smape = test()
end_time = time.time()
run_time = end_time - start_time
avg_time = run_time / obj_sum

print(run_time, avg_time)
print(mean_smape)
writer.add_scalar("loss", test_loss, 0)
writer.add_scalar("avg_time", avg_time, 0)
writer.add_scalar("smape", mean_smape, 0)

writer.close()