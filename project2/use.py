from data import SunDataset
from net import Net
import torch
path = 'D:/data/object2/class_sun'
net = Net()

net.load_state_dict(torch.load(path))
print('aa')