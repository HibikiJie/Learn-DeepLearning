import torch
from torchvision import models


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.dense_net121 = models.densenet201(pretrained=True)
        self.dense_net121.features[0] = torch.nn.Conv2d(2,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=True)
        self.dense_net121.classifier = torch.nn.Linear(1920,1024,bias=True)
        # for param in self.dense_net121.parameters():
        #     param.requires_grad = True
        self.f2 = torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,3),
            torch.nn.Softmax(dim=1),
        )

    def forward(self,x):
        x = self.dense_net121(x)
        return self.f2(x)


if __name__ == '__main__':


    net = Net()

    for param in net.dense_net121.parameters():
        if param.requires_grad:
            print('yes')
        else:print("NO")

    print(net.dense_net121.parameters())
    x = torch.randn(1,2,224,224)
    x = torch.randn(1, 2, 224, 224)
    out = net(x)
    print(out.shape)