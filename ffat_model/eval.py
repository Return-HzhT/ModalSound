import sys
sys.path.append("..")
from dataset import FfatDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from point_transformer import Net
import time
from tqdm import tqdm

torch.cuda.set_device(5)

log_dir = "./evaluation/"
if os.path.exists(log_dir):
    os.system("rm -r %s" % log_dir)
writer = SummaryWriter(log_dir)

test_dataset = FfatDataset('test')

batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Net().cuda()
model.load_state_dict(torch.load("./train/best.pth"))

loss_fn = torch.nn.MSELoss()

@torch.no_grad()
def test():
    model.eval()
    
    total_loss = 0.0
    total_smape = 0.0

    now_cnt = 0

    for data in tqdm(test_loader):
        data = data.cuda()
        ffat_pred = model(data)

        loss = loss_fn(ffat_pred, data.ffat)
        total_loss += float(loss)

        total_smape += torch.mean(torch.abs(torch.abs(ffat_pred) - torch.abs(data.ffat)) / ((torch.abs(data.ffat) + torch.abs(ffat_pred)) / 2))
        total_loss += float(loss)
        
        now_cnt += 1

    total_smape /= len(test_loader)
    total_loss /= len(test_loader)
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