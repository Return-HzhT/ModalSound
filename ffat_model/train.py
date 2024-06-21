import sys
sys.path.append("..")
from dataset import FfatDataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from point_transformer import Net

torch.cuda.set_device(3)

log_dir = "./log/"
if os.path.exists(log_dir):
    os.system("rm -r %s" % log_dir)
os.mkdir(log_dir)
writer = SummaryWriter(log_dir)

train_dataset = FfatDataset('train')
test_dataset = FfatDataset('valid')

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Net().cuda()
# model.load_state_dict(torch.load("./train/best.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

loss_fn = torch.nn.MSELoss()

def train():
    model.train()

    total_loss = 0.0
    img_flag = False
    for data in train_loader:
        data = data.cuda()
        optimizer.zero_grad()
        ffat_pred = model(data)

        loss = loss_fn(ffat_pred, data.ffat)

        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        if img_flag == False and epoch % 2 == 0:
            img_flag = True

            tmp_map_1 = ffat_pred[0].reshape(1, 64, -1)
            tmp_map_2 = data.ffat[0].reshape(1, 64, -1)

            min_value = torch.min(tmp_map_1.min(),tmp_map_2.min())
            max_value = torch.max(tmp_map_1.max(),tmp_map_2.max())

            tmp_map_1 = (tmp_map_1 - min_value) / (max_value - min_value)
            tmp_map_2 = (tmp_map_2 - min_value) / (max_value - min_value)

            tmp_map_1 = torch.clamp(tmp_map_1, 0, 1)
            tmp_map_2 = torch.clamp(tmp_map_2, 0, 1)

            writer.add_image("train_ffat_map_pred", tmp_map_1, epoch)
            writer.add_image("train_ground_truth", tmp_map_2, epoch)
    
    total_loss /= len(train_loader)
    return total_loss


@torch.no_grad()
def test():
    model.eval()

    total_loss = 0.0
    img_flag = False
    for data in test_loader:
        data = data.cuda()
        ffat_pred = model(data)

        loss = loss_fn(ffat_pred, data.ffat)
        total_loss += float(loss)

        if img_flag == False and epoch % 2 == 0:
            img_flag = True

            tmp_map_1 = ffat_pred[0].reshape(1, 64, -1)
            tmp_map_2 = data.ffat[0].reshape(1, 64, -1)

            min_value = torch.min(tmp_map_1.min(),tmp_map_2.min())
            max_value = torch.max(tmp_map_1.max(),tmp_map_2.max())

            tmp_map_1 = (tmp_map_1 - min_value) / (max_value - min_value)
            tmp_map_2 = (tmp_map_2 - min_value) / (max_value - min_value)

            tmp_map_1 = torch.clamp(tmp_map_1, 0, 1)
            tmp_map_2 = torch.clamp(tmp_map_2, 0, 1)

            writer.add_image("test_ffat_map_pred", tmp_map_1, epoch)
            writer.add_image("test_ground_truth", tmp_map_2, epoch)

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