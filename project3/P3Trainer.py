from project3.P3Dataset import Voc2012DataSet
from project3.P3YOLO_V3 import YOLOVision3Net
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm


class Trainer:

    def __init__(self, Net=YOLOVision3Net):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('准备使用设备%s训练网络' % self.device)
        self.net = Net().train()
        self.net.load_state_dict(torch.load('D:/data/object3/netparm/net.pth'))
        self.net = self.net.to(self.device)
        print('模型初始化完成')
        self.data_set = Voc2012DataSet()
        self.data_loader = DataLoader(self.data_set, 5, True, num_workers=4)
        self.binary_cross_entropy = torch.nn.BCELoss().to(self.device)
        self.mse_loss = torch.nn.MSELoss().to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss().to(self.device)
        print('损失函数初始化完成')
        self.optimizer = torch.optim.Adam(self.net.parameters())
        print('优化器初始化完成')
        self.summary_writer = SummaryWriter('D:/data/object3/logs')
        self.sigmoid = torch.nn.Sigmoid()

    def train(self):
        i = 0
        epoch = 0
        while True:
            loss_sum = 0
            for images, targets_13, targets_26, targets_52 in self.data_loader:

                images = images.to(self.device)
                targets_13 = targets_13.to(self.device)
                targets_26 = targets_26.to(self.device)
                targets_52 = targets_52.to(self.device)
                predict1, predict2, predict3 = self.net(images)
                loss1 = self.compute_loss(predict1, targets_13)
                loss2 = self.compute_loss(predict2, targets_26)
                loss3 = self.compute_loss(predict3, targets_52)
                loss = loss1 + loss2 + loss3

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.summary_writer.add_scalar('loss', loss.item(), i)
                i += 1
                loss_sum += loss.item()
                if i % 100 == 0:
                    torch.save(self.net.state_dict(), f'D:/data/object3/netparm/net.pth')
                # print(i,loss.item())
            epoch += 1
            self.summary_writer.add_scalar('loss_epoch', loss_sum / len(self.data_loader), epoch)
            print(epoch, loss_sum / len(self.data_loader))

    def compute_loss(self, predict, target):

        """标签形状为（N,H,W,3,6）"""
        mask_positive = target[:, :, :, :, 0] > 0.5
        mask_negative = target[:, :, :, :, 0] < 0.5

        target_positive = target[mask_positive]
        target_negative = target[mask_negative]
        number, _ = target_positive.shape
        predict_positive = predict[mask_positive]
        predict_negative = predict[mask_negative]

        '''置信度损失'''
        if number > 0:
            loss_c_p = self.binary_cross_entropy(self.sigmoid(predict_positive[:, 0]), target_positive[:, 0])
        else:
            loss_c_p = 0
        loss_c_n = self.binary_cross_entropy(self.sigmoid(predict_negative[:, 0]), target_negative[:, 0])
        loss_c = loss_c_n + loss_c_p

        '''边框回归'''
        if number > 0:
            loss_box1 = self.mse_loss(self.sigmoid(predict_positive[:, 1:3]), target_positive[:, 1:3])
            loss_box2 = self.mse_loss(predict_positive[:, 3:5], target_positive[:, 3:5])

            '''分类损失'''
            loss_class = self.ce_loss(predict_positive[:, 5:], target_positive[:, 5].long())
        else:
            loss_box1 = 0
            loss_box2 = 0
            loss_class = 0
        return loss_c + (loss_box1 + loss_box2) + loss_class


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
