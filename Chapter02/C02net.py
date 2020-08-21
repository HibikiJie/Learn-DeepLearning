import torch
import cv2
import os
from PIL import Image
from torchvision.transforms import ToTensor

class Net(torch.nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,1,1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64,128,3,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128,256,3,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(256,256,3,1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(256,512,3,1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512,1024,3,1),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 4)
        )


    def forward(self,x):
        return self.classifier(self.features(x).reshape(-1,1024))


if __name__ == "__main__":
    net = Net()
    # x = torch.randn(1,3,300,300)
    # print(net(x).shape)
    net.load_state_dict(torch.load('D:/data/chapter2/chapter02'))
    totensor = ToTensor()
    file_names = os.listdir('D:/data/chapter2/shuju')
    for file_name in file_names:
        img = Image.open(f'D:/data/chapter2/shuju/{file_name}')
        img_tensor = totensor(img)
        out = net(img_tensor.unsqueeze(0))
        img = cv2.imread(f'D:/data/chapter2/shuju/{file_name}')
        x1 = int(out[0][0].item()*300)
        y1 = int(out[0][1].item()*300)
        x2 = int(out[0][2].item() * 300)
        y2 = int(out[0][3].item() * 300)
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("JK",img)
        if cv2.waitKey() == ord('n'):
            continue
    cv2.destroyAllWindows()


