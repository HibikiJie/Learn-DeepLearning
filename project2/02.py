from project2.zExplorerOpenCV import Explorer
from project2.P2Net import Net
import cv2
import torch
import numpy
import os

torch.nn.Dropout()
net = Net()
# net = densenet121()
net.load_state_dict(torch.load('D:/data/object2/netParam/net18.pth'))
features = []
net.eval()
for feature in os.listdir('D:/data/object2/datas'):
    feature_data = torch.load(f'D:/data/object2/datas/{feature}').squeeze(0)
    features.append(feature_data)
features = torch.stack(features)
features = torch.nn.functional.normalize(features,dim=1)


image = cv2.imread('1.png')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_tensor = torch.from_numpy(
                        numpy.transpose(image, (2, 0, 1)) / 255 - 0.5).float().unsqueeze(0)
feature = net(image_tensor)
feature = torch.nn.functional.normalize(feature, dim=1).squeeze(0)
cos_theta = torch.matmul(features, feature)
print(cos_theta)