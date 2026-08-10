import torch
import sys
from plyfile import PlyData
import numpy as np

assert torch.cuda.is_available()


class Mlp(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, out_dim),
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
xyz = xyz[order]
print(xyz)
#sys.exit(0)

num = 1024

def load_tensor(k):
    return torch.from_numpy(v[k][order][:num]).cuda()

xyz = torch.stack(
    [
        load_tensor("x"),
        load_tensor("y"),
        load_tensor("z"),
    ],
    dim=1,
)

props = torch.stack(
    [
        load_tensor("f_dc_0"),
        load_tensor("f_dc_1"),
        load_tensor("f_dc_2"),
        load_tensor("opacity"),
    ],
    dim=1,
)

model = Mlp(props.shape[1]).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
iters = 50_000

for i in range(iters):
    optimizer.zero_grad()

    pred = model(xyz)
    loss = ((pred - props) ** 2).mean()

    loss.backward()
    optimizer.step()
    print(i, loss)