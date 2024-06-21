import sys
sys.path.append("..")
from dataset import EigenvalueDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from point_transformer import Net
import time

torch.cuda.set_device(3)

log_dir = "./evaluation/"
if os.path.exists(log_dir):
    os.system("rm -r %s" % log_dir)
writer = SummaryWriter(log_dir)

test_dataset = EigenvalueDataset('test')

batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = Net().cuda()
model.load_state_dict(torch.load("./train/best.pth"))

mask_loss_fn = torch.nn.BCELoss()

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
    f1_score = 0 if recall + precision == 0 else (2 * recall * precision) / (recall + precision)

    return total_loss, acc, recall, precision, f1_score

obj_sum = len(test_loader.dataset)
start_time = time.time()
loss, acc, recall, precision, f1_score = test()
end_time = time.time()
run_time = end_time - start_time
avg_time = run_time / obj_sum

print(run_time, avg_time)

writer.add_scalar("loss", loss, 0)
writer.add_scalar("acc", acc, 0)
writer.add_scalar("recall", recall, 0)
writer.add_scalar("precision", precision, 0)
writer.add_scalar("f1", f1_score, 0)
writer.add_scalar("avg_time", avg_time, 0)
writer.close()