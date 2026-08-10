import torch
import sys
from plyfile import PlyData
import numpy as np
from torch import nn
import IPython


assert torch.cuda.is_available()

from torch.utils.tensorboard import SummaryWriter

class BatchedLinear(nn.Module):
    def __init__(self, num_chunks, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_chunks, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_chunks, 1, out_dim))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):  # x: (num_chunks, chunk_size, in_dim)
        return torch.bmm(x, self.weight) + self.bias


class Mlp(nn.Module):
    def __init__(self, num_chunks, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            BatchedLinear(num_chunks, 3, 64),
            nn.SiLU(),
            BatchedLinear(num_chunks, 64, 64),
            nn.SiLU(),
            BatchedLinear(num_chunks, 64, 64),
            nn.SiLU(),
            BatchedLinear(num_chunks, 64, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def _split_by_3(x):
    x = x & np.uint64(0x1fffff)
    x = (x | (x << 32)) & np.uint64(0x1f00000000ffff)
    x = (x | (x << 16)) & np.uint64(0x1f0000ff0000ff)
    x = (x | (x << 8))  & np.uint64(0x100f00f00f00f00f)
    x = (x | (x << 4))  & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x << 2))  & np.uint64(0x1249249249249249)
    return x

def morton3d(xyz):
    q = np.clip(xyz * ((1 << 21) - 1), 0, (1 << 21) - 1).astype(np.uint64)

    x = _split_by_3(q[:, 0])
    y = _split_by_3(q[:, 1])
    z = _split_by_3(q[:, 2])

    return x | (y << 1) | (z << 2)

ply = PlyData.read(sys.argv[1])
v = ply["vertex"]

xyz = [v["x"], v["y"], v["z"]]

minimum = np.array([val.min() for val in xyz])
maximum = np.array([val.max() for val in xyz])
xyz = np.stack(xyz, axis=1)
rescaled = (xyz - minimum)/(maximum - minimum)
order = np.argsort(morton3d(rescaled), kind="stable")
n_points = xyz.shape[0]//16
xyz = torch.from_numpy(xyz[order][:n_points]).cuda()
print(xyz.shape, n_points)

props = torch.stack(
    [
        torch.from_numpy(v["f_dc_0"][order][:n_points]).cuda(),
        torch.from_numpy(v["f_dc_1"][order][:n_points]).cuda(),
        torch.from_numpy(v["f_dc_2"][order][:n_points]).cuda(),
        torch.from_numpy(v["opacity"][order][:n_points]).cuda()
    ],
    dim=1,
)

learning_rate = 1.5e-3
iters = 50_000

writer = SummaryWriter(f"splat_mlp/n{iters}-l{learning_rate}")

chunk_size = 1024
n_chunks = n_points // chunk_size          # drop remainder points for simplicity


xyz = xyz[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, 3)
props = props[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, props.shape[1])

model = Mlp(n_chunks, props.shape[2]).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(iters):
    optimizer.zero_grad()
    pred = model(xyz)                      # all chunks, all points, one call
    loss = ((pred - props) ** 2).mean()
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    writer.add_scalar("train/loss", loss, i)

IPython.embed()