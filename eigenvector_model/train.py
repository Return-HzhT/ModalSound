import sys
sys.path.append("..")
from dataset import EigenvectorDataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from src.visualize import CombinedFig
import cv2
from pointnet2 import Net

torch.cuda.set_device(0)

log_dir = "./log/"
if os.path.exists(log_dir):
    os.system("rm -r %s" % log_dir)
os.mkdir(log_dir)
writer = SummaryWriter(log_dir)

train_dataset = EigenvectorDataset('train')
test_dataset = EigenvectorDataset('valid')

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Net().cuda()
# model.load_state_dict(torch.load("./tmp_train/best.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

vec_loss_fn = torch.nn.MSELoss()

def train():
    model.train()

    total_loss = 0
    img_flag = False

    for data in train_loader:
        data = data.cuda()
        optimizer.zero_grad()
        vec_pred = model(data)

        mask = data.mask.bool().reshape(batch_size, 32)
        vec_pred = vec_pred.permute(1, 2, 0).reshape(3,-1).permute(1, 0)
        vec_gt = data.vecs.permute(1, 2, 0).reshape(3,-1).permute(1, 0)

        points_mask = mask[data.batch].permute(1, 0).reshape(-1)
        loss = vec_loss_fn(vec_pred[points_mask,:], vec_gt[points_mask,:])

        loss.backward()
        optimizer.step()
        total_loss += float(loss)

        if img_flag == False and epoch % 10 == 0:
            img_flag = True
            pcd_sum = data.batch.shape[0] - data.batch.count_nonzero()
            pcd = data.pos[:pcd_sum]
            vec = (vec_pred[points_mask,:]**2).sum(axis=1)
            gt_vec = (vec_gt[points_mask,:]**2).sum(axis=1)

            CombinedFig().add_points(
                pcd, vec[:pcd_sum]
            ).save("./tmp/train_tmp1.png")
            CombinedFig().add_points(
                pcd, gt_vec[:pcd_sum]
            ).save("./tmp/train_tmp2.png")

            now_fig = cv2.imread("./tmp/train_tmp1.png")[:,:,::-1].transpose((2,0,1))
            gt_fig = cv2.imread("./tmp/train_tmp2.png")[:,:,::-1].transpose((2,0,1))

            writer.add_image("train: eigenvectors_pred", now_fig, epoch)
            writer.add_image("train: eigenvectors_ground_truth", gt_fig, epoch)

    total_loss /= len(train_loader)
    return total_loss


@torch.no_grad()
def test():
    model.eval()

    total_loss = 0
    img_flag = False

    for data in test_loader:
        data = data.cuda()
        vec_pred = model(data)

        mask = data.mask.bool().reshape(batch_size, 32)
        vec_pred = vec_pred.permute(1, 2, 0).reshape(3,-1).permute(1, 0)
        vec_gt = data.vecs.permute(1, 2, 0).reshape(3,-1).permute(1, 0)
        points_mask = mask[data.batch].permute(1, 0).reshape(-1)
        loss = vec_loss_fn(vec_pred[points_mask,:], vec_gt[points_mask,:])

        total_loss += float(loss)

        if img_flag == False and epoch % 10 == 0:
            img_flag = True
            pcd_sum = data.batch.shape[0] - data.batch.count_nonzero()
            pcd = data.pos[:pcd_sum]
            vec = (vec_pred[points_mask,:]**2).sum(axis=1)
            gt_vec = (vec_gt[points_mask,:]**2).sum(axis=1)

            CombinedFig().add_points(
                pcd, vec[:pcd_sum]
            ).save("./tmp/test_tmp1.png")
            CombinedFig().add_points(
                pcd, gt_vec[:pcd_sum]
            ).save("./tmp/test_tmp2.png")

            now_fig = cv2.imread("./tmp/test_tmp1.png")[:,:,::-1].transpose((2,0,1))
            gt_fig = cv2.imread("./tmp/test_tmp2.png")[:,:,::-1].transpose((2,0,1))

            writer.add_image("test: eigenvectors_pred", now_fig, epoch)
            writer.add_image("test: eigenvectors_ground_truth", gt_fig, epoch)

    total_loss /= len(test_loader)
    return total_loss

train_loss_result = []
test_loss_result = []
best_loss = 1e10
num_epoch = 200
for epoch in tqdm(range(num_epoch)):
    train_loss = train()
    test_loss = test()
    scheduler.step()

    writer.add_scalar("train_loss", train_loss, epoch)
    train_loss_result.append(train_loss)
    writer.add_scalar("test_loss", test_loss, epoch)
    test_loss_result.append(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), "./train/best.pth")
    print(f'Epoch: {epoch:05d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

plt.plot([i for i in range(len(train_loss_result))], train_loss_result)
plt.savefig("./train/train.png")
plt.cla()
plt.plot([i for i in range(len(test_loss_result))], test_loss_result)
plt.savefig("./train/test.png")
writer.close()