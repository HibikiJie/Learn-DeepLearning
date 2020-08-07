import random
import cv2
import torch

y = torch.rand(8,4)
x = torch.rand(8,1)
print(x)
print(torch.where(x > 0.5))

mask = torch.where(x>0.5)[0]
print(y[mask])