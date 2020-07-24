import torch
from torchvision.models import densenet121


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
        print(x.shape)
        return self.fc2(x)



if __name__ == '__main__':
    net = Net()
    print(net.parameters)
    x1 = torch.randn(2, 1, 224, 224)
    x2 = torch.randn(2,10)
    print(net(x1,x2))
