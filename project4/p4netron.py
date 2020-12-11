from torchvision.models import densenet121
import torch

net = densenet121(True)

x = torch.randn(1,3,224,224)
m = torch.jit.trace(net,(x,))
m.save('densenet121.pt')
