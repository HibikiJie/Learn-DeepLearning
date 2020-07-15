import torch
from torchvision import models


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.dense_net121 = models.densenet121(pretrained=True)
        self.dense_net121.features[0] = torch.nn.Conv2d(2,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=True)
        self.dense_net121.classifier = torch.nn.Linear(1024,1024,bias=True)
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512,128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128,3),
            torch.nn.Softmax(dim=1),
        )

    def forward(self,x):
        x = self.dense_net121(x)
        x = self.layer(x)
        return x


if __name__ == '__main__':

    net = Net()
    # print(net)
    x = torch.randn(3,2,448,448)
    out = net(x)
    print(out)