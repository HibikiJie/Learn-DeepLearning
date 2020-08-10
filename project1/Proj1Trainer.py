from torch.utils.data import DataLoader
from project1.Proj1FaceDataSet import FaceDataSet
from project1.Proj1Net import PNet, RNet, ONet
from project1.Proj1NMS import non_maximum_suppression
from torch.utils.tensorboard import SummaryWriter
import torch


class Trainer(object):

    def __init__(self, net, image_size, data_path='D:/data/object1/train',logs_path='D:/data/object1/logs'):
        """
        :param net:
        :param image_size:
        :param logs_path:
        """
        '''查看是否能调用cuda'''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('正在使用%s进行训练' % self.device)

        '''实例化网络,并加载模型参数'''
        self.net = net()
        self.net.load_parameters()

        '''将网络加载至可调用硬件'''
        self.net.to(self.device)

        '''实例化数据加载集'''
        self.data_set = FaceDataSet(path=data_path,image_size=image_size)
        self.data_loader = DataLoader(self.data_set, 512, True)

        '''实例化summary_writer（摘要作家）'''
        self.summary_writer = SummaryWriter(logs_path)

        '''创建损失函数'''
        self.bce_loss = torch.nn.BCELoss().to(self.device)  # 训练置信度的损失函数（二分类）
        self.mse_loss = torch.nn.MSELoss().to(self.device)  # 训练偏移量的损失函数

        '''创建优化器'''
        self.optimizer = torch.optim.Adam(self.net.parameters())
        # self.optimizer = torch.optim.SGD(self.net.parameters(),lr=0.00001)

        print('训练器初始化完成')

    def train(self):
        count = 0
        print('训练开始')
        while True:
            for image, confidence, coordinate in self.data_loader:
                image = image.to(self.device)
                confidence_target = confidence.to(self.device)
                coordinate_target = coordinate.to(self.device)
                out_confidence, out_coordinate = self.net(image)

                '''变换形状，使得P网络能够计算损失'''
                out_confidence = out_confidence.reshape(-1, 1)
                out_coordinate = out_coordinate.reshape(-1, 4)

                '''计算置信度的损失'''
                category_mask = torch.where(confidence < 1.5)[0]
                cls_loss = self.bce_loss(out_confidence[category_mask], confidence_target[category_mask])

                '''计算bound回归的损失'''
                offset_mask = torch.where(confidence > 0.5)[0]
                offset_loss = self.mse_loss(out_coordinate[offset_mask], coordinate_target[offset_mask])

                '''计算总损失'''
                loss = cls_loss + offset_loss

                '''反向传播'''
                self.optimizer.zero_grad()  # 清空之前的梯度
                loss.backward()  # 反向传播求梯度
                self.optimizer.step()  # 优化网络参数

                loss_txt = loss.detach().cpu().item()

                '''记录损失'''
                self.summary_writer.add_scalar("loss", loss_txt, count)
                print(count,loss_txt)
                '''保存数据'''
                if count % 100 == 0:
                    self.net.save_parameters()
                count += 1


if __name__ == '__main__':
    trainer = Trainer(ONet, '48')
    trainer.train()
