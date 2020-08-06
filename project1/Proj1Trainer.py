from torch.utils.data import DataLoader
from project1.Proj1FaceDataSet import FaceDataSet
from project1.Proj1Net import PNet, RNet, ONet
from project1.Proj1NMS import non_maximum_suppression
from torch.utils.tensorboard import SummaryWriter
import torch


class Trainer(object):

    def __init__(self, logs_path='D:/data/object1/logs'):
        self.data_set = FaceDataSet()
        self.summary_writer = SummaryWriter('D:/data/object1/logs')

    def train(self):
        pass