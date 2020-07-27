from Chapter02.C02net import Net
from Chapter02.C02data import DataYellow
from torch.utils.data import DataLoader
import torch,os
import tqdm

class Train():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.dataset = DataYellow()
        self.data_loader = DataLoader(self.dataset,100,True)

        self.net = Net().to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(),0.0001)
        self.loss_function = torch.nn.MSELoss().to(self.device)
        if os.path.exists('D:/data/chapter2/chapter02'):
            self.net.load_state_dict(torch.load('D:/data/chapter2/chapter02'))


    def __call__(self):
        print('训练开始')
        for epoch in range(100):
            loss_sum = 0
            for img,target in tqdm.tqdm(self.data_loader):

                out = self.net(img.to(self.device))
                loss = self.loss_function(out,target.to(self.device))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.cpu().detach().item()
            print(epoch,loss_sum/len(self.data_loader))
            torch.save(self.net.state_dict(),'D:/data/chapter2/chapter02')

if __name__ == '__main__':
    train = Train()
    train()