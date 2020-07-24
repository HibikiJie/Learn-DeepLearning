from torch.utils.data import DataLoader,sampler
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
        self.train_data_loader = DataLoader(self.train_dataset,10,True)
        print('数据加载成功')
        self.veri_dataset = SunDataset(is_train=False)
        self.veri_data_loader = DataLoader(self.veri_dataset,5,False)

        self.net = Net().to(self.device)
        print('模型加载成功')
        self.lossf = torch.nn.BCELoss().to(self.device)

        # self.optim = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.optim = torch.optim.Adam(self.net.parameters())
        # self.summarywriter = SummaryWriter('D:\data\object2\logs')

    def __call__(self):
        train_lenth = len(self.train_data_loader)

        vari_lenth = len(self.veri_data_loader)
        print('训练开始')
        for epoch in range(100000):

            train_loss_sum = 0
            # bar = tqdm.tqdm(range(train_lenth))
            for img,taget in iter(self.train_data_loader):

                img = img.to(self.device)
                taget = taget.to(self.device)
                for i in range(100):

                    out = self.net(img)
                    loss = self.lossf(out,taget)

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    train_loss_sum += loss.detach().cpu().item()
                    # bar.update(1)
                    print(i,loss.detach().item())
            train_loss = train_loss_sum/train_lenth
            # bar.close()
            del img,taget

            '''验证'''
            veri_loss_sum = 0
            score = 0
            bar = tqdm.tqdm(range(vari_lenth))
            for img,taget in self.veri_data_loader:
                img = img.to(self.device)
                taget = taget.to(self.device)
                out = self.net(img)
                loss = self.lossf(out,taget)
                bar.update(1)
                veri_loss_sum += loss.detach().cpu().item()
                pre_tage = torch.argmax(out.cpu(), 1)
                taget_tage = torch.argmax(taget.cpu(), 1)
                score += torch.sum(torch.eq(pre_tage, taget_tage).float())
            bar.close()

            score = score/len(self.veri_dataset)

            veri_loss = veri_loss_sum/vari_lenth
            print(epoch,train_loss,veri_loss,score)
            # self.summarywriter.add_scalars("Loss",{'train':train_loss,'Verification':veri_loss},epoch)
            # self.summarywriter.add_scalar("Score",score,epoch)
            torch.save(self.net.state_dict(),'class_sun1')
            time.sleep(1)




train = Train()
train()