import torch
from torchvision.models import densenet121
from torch import jit
import cv2
import numpy

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.cnn = densenet121(True)
        self.cnn.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn.classifier = torch.nn.Linear(1024, 128, True)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(10, 128),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        h = self.cnn(x1)
        x2 = self.fc1(x2)
        x = torch.cat([h,x2],dim=1)
        # print(x.shape)
        return self.fc2(x)



if __name__ == '__main__':
    yes = 0
    no = 0

    model = Net()
    model.load_state_dict(torch.load('SunStorm_net'))
    with open('test_input_para.txt','r') as f:
        file_data = f.read().split()
    a = len(file_data)//11
    file_name = []
    file_para = []
    f = open('out.txt','w')
    for i in range(a):
        file_name.append(file_data[i*11])
        file_para.append([file_data[11*i+1:11*i+11]])
    for img_name,para in zip(file_name,file_para):
        img = cv2.imread(f'D:/sun/test_jpg_input/{img_name}',0)
        img = cv2.resize(img,(224,224))/255
        img = torch.tensor([[numpy.array(img,dtype=numpy.float32)]])
        para = torch.tensor(numpy.array(para,dtype=numpy.float32))

        out = model(img,para)
        print(out.item())
        if out.item() > 0.5:
            yes +=1
            outtt = '1'
            f.write('{}  {}\n'.format(img_name,outtt))
        else:
            no += 1
            outtt = '0'
            f.write('{}  {}\n'.format(img_name, outtt))
        print('{}  {}'.format(img_name,outtt))

    print(yes,no)