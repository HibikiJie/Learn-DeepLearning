from project1.Proj1Net import PNet, RNet, ONet
from project1.Proj1NMS import non_maximum_suppression
from torchvision.transforms import ToTensor
from time import time
import torch
from PIL import Image, ImageDraw
import numpy
import os
import cv2


class Explorer:

    def __init__(self):
        '''设置网络参数'''
        self.p_confidence_threshold = 0.8  # 建议值0.6
        self.p_nms_threshold = 0.6  # 建议值0.6
        self.r_confidence_threshold = 0.99  # 建议值0.6
        self.r_nms_threshold = 0.5  # 建议值0.5
        self.o_confidence_threshold = 0.9999  # 建议值0.9999
        self.o_nms_threshold = 0.4  # 建议值0.7

        '''实例化网络'''
        self.p_net = PNet()
        self.r_net = RNet()
        self.o_net = ONet()

        '''加载网络参数'''
        self.p_net.load_parameters()
        self.r_net.load_parameters()
        self.o_net.load_parameters()

        '''实例化图片转行器'''
        self.to_tensor = ToTensor()

    def explore(self, image):
        start_time = time()
        boxes_p = self.p_net_explore(image)
        print('p:', time() - start_time)
        print(boxes_p.shape)
        if boxes_p.shape[1] == 0:
            return numpy.array([])
        # return boxes_p.numpy()
        boxes_r = self.r_net_explore(image, boxes_p)

        print('r:', time() - start_time)
        if boxes_r.shape[1] == 0:
            return numpy.array([])
        boxes = self.o_net_explore(image, boxes_r)
        print('o:', time() - start_time)

        if boxes.shape[1] == 0:
            return numpy.array([])
        print(boxes)
        return boxes.numpy()

    def p_net_explore(self, image):
        """
        使用P网络对图像进行探索任务，
        :param image:
        :return: 返回的是经过非极大值抑制之后的对应于原图的目标框
        """
        image = image
        w, h = image.size
        boxes = []
        min_side_len = min(w, h)
        zoom_ratio = 1
        starttime = time()
        while min_side_len >= 18:
            '''图形数据升维，因为网络有批次概念'''
            image_tensor = self.to_tensor(image) - 0.5
            image_tensor = image_tensor.unsqueeze(0)
            '''
            p网络输出的置信度的形状为(1,1,h`,w`)
            偏移量的形状为：(1,4,h`,w`)    h`和w`并非原图的尺寸大小，为网络输出的特征图大小
            '''
            confidence, offset = self.p_net(image_tensor)
            confidence = confidence[0][0].detach()
            offset = offset[0].detach()
            offset = offset.permute(1, 2, 0)
            mask = torch.where(confidence > self.p_confidence_threshold)
            offset = offset[mask]
            confidence = confidence[mask].unsqueeze(1)

            '''计算位于特征图上的索引'''
            h_index = mask[0]
            w_index = mask[1]
            xy_index = torch.stack((w_index, h_index), 1)
            '''反算候选框对应于原图的位置'''
            _xy1_ = xy_index.float() * 2 / zoom_ratio
            _xy2_ = (xy_index.float() * 2 + 12) / zoom_ratio
            xy = torch.cat((_xy1_, _xy2_), 1)
            side_len = 12 / zoom_ratio
            coordinate = xy + offset * side_len
            boxes.append(torch.cat((coordinate, confidence), 1))
            '''变换图形尺寸'''
            zoom_ratio *= 0.7071068
            _w = int(w * zoom_ratio)
            _h = int(h * zoom_ratio)
            image = image.resize((_w, _h), Image.ANTIALIAS)
            min_side_len = min(_w, _h)
        print('P网络监测时间：', time() - starttime)

        return non_maximum_suppression(torch.cat(boxes, 0), self.p_nms_threshold)

    def r_net_explore(self, image, boxes):
        images, boxes_origin, side_lens = self.cut_out_image(image, boxes, 24)
        confidence, coordinate = self.r_net(images)
        confidence = confidence.detach()
        coordinate = coordinate.detach()
        '''反算'''
        index = torch.where(confidence > self.r_confidence_threshold)  # 取出大于置信度阈值的索引

        confidence = confidence[index[0]]
        boxes_origin = boxes_origin[index[0]]
        side_lens = side_lens[index[0]]
        coordinate = coordinate[index[0]]
        coordinate_ = boxes_origin + side_lens * coordinate
        boxes = torch.cat((coordinate_, confidence), 1)
        return non_maximum_suppression(boxes, self.r_nms_threshold)

    def o_net_explore(self, image, boxes):
        images, boxes_origin, side_lens = self.cut_out_image(image, boxes, 48)
        confidence, coordinate = self.o_net(images)
        confidence = confidence.detach()
        coordinate = coordinate.detach()
        '''反算'''
        index = torch.where(confidence > self.o_confidence_threshold)  # 取出大于置信度阈值的索引

        confidence = confidence[index[0]]
        boxes_origin = boxes_origin[index[0]]
        side_lens = side_lens[index[0]]
        coordinate = coordinate[index[0]]
        coordinate_ = boxes_origin + side_lens * coordinate
        boxes = torch.cat((coordinate_, confidence), 1)

        return non_maximum_suppression(boxes, self.o_nms_threshold, True)

    def cut_out_image(self, image, boxes, size):
        boxes = boxes[:, 0:4]
        w = boxes[:, [2]] - boxes[:, [0]]
        h = boxes[:, [3]] - boxes[:, [1]]
        center_x = boxes[:, [0]] + w // 2
        center_y = boxes[:, [1]] + h // 2
        center = torch.cat((center_x, center_y), 1)
        side_lens = torch.max(w, h)
        xy1 = center - side_lens // 2
        xy2 = xy1 + side_lens
        xys = torch.cat((xy1, xy2), 1)
        images = []
        for xy in xys:
            xy = tuple(xy.numpy())
            image_crop = image.crop(xy)
            image_resize = image_crop.resize((size, size))
            image_tensor = self.to_tensor(image_resize) - 0.5
            images.append(image_tensor)
        if len(images) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        return torch.stack(images), xys, side_lens


if __name__ == '__main__':

    explorer = Explorer()
    video_capture = cv2.VideoCapture("C:/Users/lieweiai/Desktop/1c478be6413f6c0c232d21f5cae9b853.mp4")

    while True:
        success, img = video_capture.read()
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_)
        boxes = explorer.explore(image)
        if not boxes.shape[0] == 0:
            cors = boxes[:, 0:4]
            for cor in cors:
                img = cv2.rectangle(img, (cor[0],cor[1]), (cor[2],cor[3]), color=(0, 0, 255), thickness=2)
        cv2.imshow('Jk', img)
        if cv2.waitKey(1) == ord('q'):
            break
