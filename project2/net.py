import torch
from torchvision import models


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.dense_net121 = models.densenet201(pretrained=True)
        self.dense_net121.features[0] = torch.nn.Conv2d(2,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=True)
        self.dense_net121.classifier = torch.nn.Linear(1920,3,bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,x):
        return self.softmax(self.dense_net121(x))


if __name__ == '__main__':
    net = Net()
    print(net.dense_net121.parameters)
    x = torch.randn(3, 2, 224, 224)
    out = net(x)
    print(out)