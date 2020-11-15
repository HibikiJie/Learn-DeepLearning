import torch
f = torch.tensor([1,2,1], dtype=torch.float32)
print(f[:, None])
f = f[:, None] * f[None, :]
print(f)
f = f[None, None]
f = f / f.sum()
print(f)