import torch
import numpy
import cv2
from torch import nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import data
import tqdm
import time


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)

        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)

        self.conv11 = nn.Conv2d(512, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.fc4 = nn.Linear(20, 128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128,128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 1)
        if os.path.exists('Networkparameters2'):
            self.load_state_dict(torch.load('Networkparameters2'))
            print("加载网络模型参数成功")

    def forward(self, x1, x2):
        x1 = torch.tanh(self.conv1(x1))
        x1 = F.max_pool2d(torch.tanh(self.conv2(x1)), (2, 2))

        x1 = torch.tanh(self.conv3(x1))
        x1 = F.max_pool2d(torch.tanh(self.conv4(x1)), (2, 2))

        x1 = torch.tanh(self.conv5(x1))
        x1 = torch.tanh(self.conv6(x1))
        x1 = F.max_pool2d(torch.tanh(self.conv7(x1)), (2, 2))

        x1 = torch.tanh(self.conv8(x1))
        x1 = torch.tanh(self.conv9(x1))
        x1 = F.max_pool2d(torch.tanh(self.conv10(x1)), (2, 2))

        x1 = torch.tanh(self.conv11(x1))
        x1 = torch.tanh(self.conv12(x1))
        x1 = F.max_pool2d(torch.tanh(self.conv13(x1)), (2, 2))

        x1 = x1.view(-1, 512)
        x1 = torch.tanh(self.fc1(x1))
        x1 = torch.tanh(self.fc2(x1))
        x1 = torch.tanh(self.fc3(x1))
        x = torch.cat((x1, x2), 1)
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = torch.sigmoid(self.fc8(x))
        return x

class BCEFocalLoss(torch.nn.Module):
    def __init__(self,gamma=2,alpha=0.25):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,y_pred,target):
        loss = -20*(1-y_pred)**self.gamma*target*torch.log(y_pred)-(1-self.alpha)*y_pred**self.gamma*(1-target)*torch.log(1-y_pred)
        loss = torch.mean(loss)
        return loss





net = Net().to(device)
print(net.conv1.weight)
exit()
optim = torch.optim.Adam(net.parameters(), lr=0.0001)
lossf = BCEFocalLoss().to(device)

sun_dataset = data.SunDataset()
train_load = DataLoader(sun_dataset, 35, True)
for i in range(10000):
    print("下一轮训练开始")

    lo=0
    count=0
    for data, para, taget in tqdm.tqdm(train_load):
        data = data.type(torch.FloatTensor).to(device)
        para = para.type(torch.FloatTensor).to(device)
        taget = taget.type(torch.FloatTensor).to(device)
        optim.zero_grad()
        out = net(data, para)
        loss = lossf(out, taget)
        loss.backward()
        optim.step()
        print(out.detach())
        a = loss.item()
        lo += a
        count += 1
        # print(i,lo/count)
    torch.save(net.state_dict(), 'Networkparameters2')
    print("参数保存成功")



"========================================"
# class SunDataset(Dataset):
#
#     def __init__(self,root="D:/sun/",is_train=True):
#         self.dataset = []
#         img_dir = "train" if is_train else "test"
#         file1 = open("test_input_para.txt")
#         para_in = file1.read().split()
#         a = len(para_in)//11
#         for i in range(a):
#             image_path = "D:/sun/test/"+para_in[11*i]
#             self.dataset.append((image_path,para_in[11*i+1:11*i+11]))
#
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         data = self.dataset[index]
#         file_name = data[0].split("/")[-1]
#         img_data = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE)
#         img_data = img_data.reshape((1,1,224,224))
#         img_data = img_data/255
#         para = data[1]
#         para = numpy.array(para,numpy.float64).reshape((1,10))
#         img_data = torch.from_numpy(img_data)
#         para = torch.from_numpy(para)
#
#         return img_data,para,file_name
#
# sun_data = SunDataset()
#
# net = Net()
# a = len(sun_data)
# yes = 0
# no = 0
# file = open("out.txt","w")
# for i in range(a):
#
#     data = sun_data[i][0].type(torch.FloatTensor)
#     para = sun_data[i][1].type(torch.FloatTensor)
#     # print(data.shape)
#     out = net(data,para)
#     print(out.item())
#     if out.item() >= 0.6:
#         out = "1"
#         yes+=1
#     else:
#         out = "0"
#         no += 1
#     strss = sun_data[i][2]+"  "+out+'\n'
#     print(out)
#     file.write(strss)
#     print(strss)
#
# print(yes,no)