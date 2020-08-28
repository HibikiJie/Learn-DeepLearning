from project2.P2DataSet import FCDataSet
from project2.P2Net import Net, ArcFace
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Train:

    def __init__(self, is_cuda=True):
        self.devise = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        print('准备启用%s设备训练' % self.devise)
        self.data_set = FCDataSet()
        self.data_loader = DataLoader(self.data_set, 64, True)
        self.net = Net().train().to(self.devise).train()
        self.arc_face = ArcFace(1000, 13).to(self.devise)
        self.optimzer = torch.optim.Adam([{'params': self.net.parameters()}, {'params': self.arc_face.parameters()}])
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.devise)
        self.summary_writer = SummaryWriter('D:/data/object2/logs')

    def train(self):
        print('开始训练')
        for epoch in range(10000000):
            cos_thetas = []
            targets = []
            loss_sum = 0
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

                cos_thetas.append(cos_theta)
                targets.append(target.detach().cpu())
            cos_thetas = torch.cat(cos_thetas, dim=0)
            targets = torch.cat(targets, dim=0).unsqueeze(dim=1)
            loss_sum = (loss_sum-60)*1000
            cos_thetas = cos_thetas.gather(dim=1, index=targets)*10
            print(f'{epoch} : loss_sum:{loss_sum};cos_theta:{cos_thetas.max()},{cos_thetas.min()}')
            self.summary_writer.add_scalar('Loss',loss_sum,epoch)
            torch.save(self.net.state_dict(), f'D:/data/object2/netParam/net{epoch}.pth')
