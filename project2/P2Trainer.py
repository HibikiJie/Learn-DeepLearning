from project2.P2DataSet import FCDataSet2
from project2.P2Net import Net, ArcFace
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Train:

    def __init__(self, is_cuda=True):
        self.devise = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        print('准备启用%s设备训练' % self.devise)
        self.data_set = FCDataSet2()
        self.data_loader = DataLoader(self.data_set, 75, True)
        self.net = Net()
        self.net.load_state_dict(torch.load('D:/data/object2/netParam/net19.pth'))
        self.net = self.net.to(self.devise).train()
        self.arc_face = ArcFace(1000, 10000)
        self.arc_face.load_state_dict(torch.load('D:/data/object2/netParam/Arc19.pth'))
        self.arc_face = self.arc_face.to(self.devise)
        self.optimzer = torch.optim.Adam([{'params': self.net.parameters()}, {'params': self.arc_face.parameters()}])
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.devise)
        self.summary_writer = SummaryWriter('D:/data/object2/logs')
        print(len(self.data_loader))

    def train(self):
        print('开始训练')
        n = 0
        for epoch in range(10000000):
            loss_sum = 0
            cos_theta_mean = 0
            cos_theta_min = 1
            cos_theta_max = 0
            i = 1
            for image, target in self.data_loader:
                image = image.to(self.devise)
                target = target.to(self.devise)

                feature = self.net(image)
                out, cos_theta = self.arc_face(feature)
                loss = self.loss_function(out, target)

                self.optimzer.zero_grad()
                loss.backward()
                self.optimzer.step()

                loss_sum += loss.detach().item()
                index = target.detach().cpu().unsqueeze(1)
                cos_thetas = cos_theta.gather(dim=1, index=index) * 10
                mean = cos_thetas.mean().item()
                cos_theta_mean += mean
                cos_theta_min = cos_thetas.min().item()
                cos_theta_max = max(cos_theta_max, cos_thetas.max().item())
                a = loss_sum / i
                i += 1
                print(f'{epoch} : loss_sum:{a};cos_theta:{mean}, {cos_theta_min}, {cos_theta_max}')
                self.summary_writer.add_scalar('Loss', a, n)
                self.summary_writer.add_scalar('mean', mean, n)
                self.summary_writer.add_scalar('min', cos_theta_min, n)
                self.summary_writer.add_scalar('max', cos_theta_max, n)
                n += 1
                if i % 1000 == 0:
                    torch.save(self.net.state_dict(), f'D:/data/object2/netParam/net{epoch}.pth')
                    torch.save(self.arc_face.state_dict(), f'D:/data/object2/netParam/Arc{epoch}.pth')
            cos_theta_mean = cos_theta_mean / len(self.data_loader)
            print(f'{epoch} : loss_sum:{loss_sum};cos_theta:{cos_theta_mean}, {cos_theta_min}, {cos_theta_max}')
            torch.save(self.net.state_dict(), f'D:/data/object2/netParam/net{epoch}.pth')
            torch.save(self.arc_face.state_dict(), f'D:/data/object2/netParam/Arc{epoch}.pth')


if __name__ == '__main__':
    train = Train()
    train.train()
