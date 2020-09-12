from project3.P3YOLO_V3 import YOLOVision3Net
import torch
import numpy

class Explorer:

    def __init__(self):
        self.threshold = 0.5
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = YOLOVision3Net(out_channels=84).eval()
        self.net = self.net.to(self.device)

    def explore(self, input_):
        input_ = input_.to(self.device)
        predict1, predict2, predict3 = self.net(input_)

    def select(self, predict):
        n, c, h, w = predict.shape
        predict.permute(0, 2, 3, 1)
        predict = predict.reshape(n, h, w, 3, -1)
        mask = predict > self.threshold


    def nms_with_category(self, boxes):
        pass
