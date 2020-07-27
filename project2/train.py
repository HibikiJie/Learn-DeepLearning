from torch.utils.data import DataLoader, sampler
import torch
from torch.utils.tensorboard import SummaryWriter
from data import SunDataset
from net import Net
import tqdm
import time


class Train(object):

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.train_dataset = SunDataset()
        self.train_data_loader = DataLoader(self.train_dataset, 10, True)
        print('数据加载成功')
        self.net = Net().to(self.device)
        print('模型加载成功')
        self.lossf = torch.nn.BCELoss().to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.000001)

    def __call__(self):
        train_lenth = len(self.train_data_loader)
        print('训练开始')
        for epoch in range(100000):

            train_loss_sum = 0
            for img, taget in self.train_data_loader:
                img = img.to(self.device)
                taget = taget.to(self.device)

                out = self.net(img)
                loss = self.lossf(out, taget)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss_sum += loss.detach().cpu().item()
            train_loss = train_loss_sum / train_lenth
            print(epoch, train_loss)
            torch.save(self.net.state_dict(), 'class_sun')
            time.sleep(1)


train = Train()
train()
