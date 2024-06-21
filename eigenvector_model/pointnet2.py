import torch
from torch_geometric.nn import MLP, knn_interpolate, PointNetConv, fps, global_max_pool, radius

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
    

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 128, 128, 256]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([256 + 3, 512, 512, 1024]))
        self.sa3_module = GlobalSAModule(MLP([1024 + 3, 512, 1024, 1024, 2048]))

        self.fp3_module = FPModule(1, MLP([2048 + 1024, 1024, 512, 512]))
        self.fp2_module = FPModule(3, MLP([512 + 256, 256, 128, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 256, 512, 256, 256]))

        self.mlp = MLP([8, 32, 128, 32, 8, 3], dropout=0.5, norm=None)

        # xavier
        for m in self.modules():
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if len(param.shape) < 2:
                        torch.nn.init.xavier_normal_(param.unsqueeze(0))
                    else:
                        torch.nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    def forward(self, data):
        x, pos, batch = data.pos, data.pos, data.batch

        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = x.reshape(-1, 32, 8)
        vec_pred = self.mlp(x).permute(0,2,1).reshape(-1,3,32)

        return vec_pred
