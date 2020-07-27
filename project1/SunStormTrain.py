from SunStormDataset import SunStormDataset
from SunStormNet import Net
import torch
from torch.utils.data import DataLoader
import os
import tqdm

# print('模块导入成功')
"#########################################################"
batch_size = 20
lr = 0.000000001




'''#######################################################'''

class Train():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        '''初始化数据加载器'''
        self.dataset = SunStormDataset()
        self.data_loader = DataLoader(self.dataset, batch_size, sampler=self.dataset.simpler,num_workers=4)
        '''初始化网络'''
        print('初始化网络')
        self.net = Net().to(self.device)
        print('网络初始化成功')
        self.opttim = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.loss_function = torch.nn.BCELoss().to(self.device)
        if os.path.exists('SunStorm_net'):
            self.net.load_state_dict(torch.load('SunStorm_net'))
            print('模型参数加载成功')

    def __call__(self):
        print('开始训练')
        for epoch in range(100000):
            loss_sum = 0

            for img, para, target in self.data_loader:
                print(target.sum())
                img = img.to(self.device)
                para = para.to(self.device)
                target = target.to(self.device)
                out = self.net(img, para)
                loss = self.loss_function(out, target)

                self.opttim.zero_grad()
                loss.backward()
                self.opttim.step()
                # print(loss.item())
                loss_sum += loss.item()

            print(epoch, loss_sum / len(self.data_loader))
            torch.save(self.net.state_dict(), 'SunStorm_net')


if __name__ == '__main__':
    train = Train()
    train()
