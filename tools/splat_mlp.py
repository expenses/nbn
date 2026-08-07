import torch
import sys
from plyfile import PlyData
from torch.utils.tensorboard import SummaryWriter

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

ply = PlyData.read(sys.argv[1])

num = 1024

def load_tensor(k):
    return torch.from_numpy(v[k][:num]).cuda()

v = ply["vertex"]

xyz = torch.stack([
    load_tensor("x"),
    load_tensor("y"),
    load_tensor("z"),
], dim=1)

props = torch.stack([
    load_tensor("f_dc_0"),
    load_tensor("f_dc_1"),
    load_tensor("f_dc_1"),
    load_tensor("opacity"),
], dim=1)

print(xyz)

learning_rate = 0.001

for r in range(20):
    iters = 50_001

    model = Mlp(props.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(f"splat_mlp/n{iters}-l{learning_rate}")

    for i in range(iters):
        pred = model(xyz)
        loss = ((pred - props) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss, i)

    learning_rate *= 1.5
    #learning_rate = 1e-4
