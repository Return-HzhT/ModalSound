import sys
sys.path.append("..")
from dataset import EigenvalueDataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from point_transformer import Net

torch.cuda.set_device(1)

log_dir = "./log"
if os.path.exists(log_dir):
    os.system("rm -r %s" % log_dir)
os.mkdir(log_dir)
writer = SummaryWriter(log_dir)


train_dataset = EigenvalueDataset('train')
test_dataset = EigenvalueDataset('valid')

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = Net().cuda()
# model.load_state_dict(torch.load("./train/best.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

mask_loss_fn = torch.nn.BCELoss()
def train():
    model.train()

    total_loss = 0
    acc = 0.0
    tp = 0
    gt_p = 0
    pred_p = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.cuda()
        mask_pred = model(data)
        mask_gt = data.mask.reshape(-1, 32)

        loss = mask_loss_fn(mask_pred, mask_gt)

        predict = torch.where(mask_pred>0.5,1,0)
        acc += (predict==mask_gt).sum().item()
        tp += torch.logical_and(predict==1, mask_gt==1).sum().item()
        gt_p += (mask_gt==1).sum().item()
        pred_p += (predict==1).sum().item()

        loss.backward()
        optimizer.step()
        total_loss += float(loss)

    total_loss /= len(train_loader)
    acc /= 32 * len(train_loader.dataset)
    recall = 0 if gt_p == 0 else tp / gt_p
    precision = 0 if pred_p == 0 else tp / pred_p

    return total_loss, acc, recall, precision


@torch.no_grad()
def test():
    model.eval()

    total_loss = 0
    acc = 0.0
    tp = 0
    gt_p = 0
    pred_p = 0

    for data in test_loader:
        data = data.cuda()
        mask_pred = model(data)
        mask_gt = data.mask.reshape(-1, 32)

        loss = mask_loss_fn(mask_pred, mask_gt)

        predict = torch.where(mask_pred>0.5,1,0)
        acc += (predict==mask_gt).sum().item()
        tp += torch.logical_and(predict==1, mask_gt==1).sum().item()
        gt_p += (mask_gt==1).sum().item()
        pred_p += (predict==1).sum().item()

        total_loss += float(loss)

    total_loss /= len(test_loader)
    acc /= 32 * len(test_loader.dataset)
    recall = 0 if gt_p == 0 else tp / gt_p
    precision = 0 if pred_p == 0 else tp / pred_p

    return total_loss, acc, recall, precision

train_loss_result = []
test_loss_result = []
best_loss = 1e10
num_epoch = 200
for epoch in tqdm(range(num_epoch)):
    train_loss, train_acc, train_recall, train_precision = train()
    test_loss, test_acc, test_recall, test_precision = test()
    scheduler.step()

    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("train_acc", train_acc, epoch)
    writer.add_scalar("test_acc", test_acc, epoch)
    writer.add_scalar("train_recall", train_recall, epoch)
    writer.add_scalar("test_recall", test_recall, epoch)
    writer.add_scalar("train_precision", train_precision, epoch)
    writer.add_scalar("test_precision", test_precision, epoch)

    train_loss_result.append(train_loss)
    test_loss_result.append(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), "./train/best.pth")
    print(f'Epoch: {epoch:05d}\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Recall: {train_recall:.4f}, Train Precision: {train_precision:.4f}\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Recall: {test_recall:.4f}, Test Precision: {test_precision:.4f}')

plt.plot([i for i in range(len(train_loss_result))], train_loss_result)
plt.savefig("./train/train.png")
plt.cla()
plt.plot([i for i in range(len(test_loss_result))], test_loss_result)
plt.savefig("./train/test.png")
writer.close()