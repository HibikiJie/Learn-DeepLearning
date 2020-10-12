from project3.P3YOLO_V3 import YOLOVision3Net
from project3.P3Set import Set
import torch
import numpy
import cv2
from time import time
import os
class Explorer:

    def __init__(self, is_cuda=False):
        self.set = Set()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.device = torch.device('cpu')
        self.net = YOLOVision3Net(out_channels=84)
        self.net.load_state_dict(torch.load('D:/data/object3/netparm/net1008.pth'))
        self.net.eval()

    def explore(self, input_):
        start = time()
        input_ = torch.from_numpy(input_).float() / 255
        input_ = input_.permute(2, 0, 1).unsqueeze(0)
        input_ = input_.to(self.device)  # 数据传入处理设备

        '''模型预测'''
        predict1, predict2, predict3 = self.net(input_)
        print('cost_time1:', time() - start)
        boxes1, category1 = self.select(predict1.detach().cpu().numpy(), 32, 13)
        boxes2, category2 = self.select(predict2.detach().cpu().numpy(), 16, 26)
        boxes3, category3 = self.select(predict3.detach().cpu().numpy(), 8, 52)
        boxes = numpy.vstack((boxes1, boxes2, boxes3))
        category = numpy.hstack((category1, category2, category3))
        boxes, category = self.nms_with_category(boxes, category)
        print('cost_time2:',time()-start)
        return boxes, category

    def select(self, predict, len_side, fm_size):
        """
        通过阈值筛选，并且完成反算。传输参数为array（N,C,H,W）.

        参数：
            predict: 预测值
        返回:

        """
        n, h, w, c, _ = predict.shape

        '''形状变换'''

        '''挑选置信度大于阈值的数据,获取位置索引'''
        predict[:, :, :, :, 0] = self.sigmoid(predict[:, :, :, :, 0])
        index = numpy.where(predict[:, :, :, :, 0] > self.set.threshold)

        '''获取宽高，框类索引'''
        index_h = index[1]
        index_w = index[2]
        box_base = index[3]

        '''通过索引，索引出数据'''
        boxes_with_category = predict[index]

        '''反算回原图'''
        c_x = (index_w + self.sigmoid(boxes_with_category[:, 1])) * len_side
        c_y = (index_h + self.sigmoid(boxes_with_category[:, 2])) * len_side

        w = numpy.exp(boxes_with_category[:, 3]) * self.set.boxes_base2[fm_size][box_base][:, 0]
        h = numpy.exp(boxes_with_category[:, 4]) * self.set.boxes_base2[fm_size][box_base][:, 1]

        x1 = c_x - w / 2
        y1 = c_y - h / 2

        x2 = x1 + w
        y2 = y1 + h

        '''计算所属类别'''
        category = boxes_with_category[:, 5:]
        category = numpy.argmax(category, axis=1)

        '''返回边框和类别信息'''
        return numpy.stack((boxes_with_category[:, 0], x1, y1, x2, y2), axis=1), category

    def nms_with_category(self, boxes, categorys):
        if boxes.size == 0:
            return numpy.array([]), numpy.array([])
        """根据类别的不同，进行非极大值抑制"""
        picked_boxes = []
        picked_category = []
        for category_index in range(self.set.num_category):
            '''索引该类别的数据'''
            index = categorys == category_index
            box1 = boxes[index]
            '''排序'''
            order = numpy.argsort(box1[:, 0])[::-1]
            box1 = box1[order]

            while box1.shape[0] > 1:
                max_box = box1[0]
                picked_boxes.append(max_box)
                picked_category.append(numpy.array([category_index]))
                left_box = box1[1:]
                iou = self.calculate_iou(max_box, left_box)
                index = iou < self.set.iou_threshold
                box1 = left_box[index]
            if box1.shape[0] > 0:
                picked_boxes.append(box1[0])
                picked_category.append(numpy.array([category_index]))

        return numpy.vstack(picked_boxes), numpy.hstack(picked_category)

    def calculate_iou(slef, box1, box2):
        area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
        area2 = (box2[:, 3] - box2[:, 1]) * (box2[:, 4] - box2[:, 2])

        x1 = numpy.maximum(box1[1], box2[:, 1])
        y1 = numpy.maximum(box1[2], box2[:, 2])
        x2 = numpy.minimum(box1[3], box2[:, 3])
        y2 = numpy.minimum(box1[4], box2[:, 4])

        intersection_area = numpy.maximum(0, x2 - x1) * numpy.maximum(
            0, y2 - y1)
        return intersection_area / numpy.minimum(area1, area2)

    def sigmoid(self, x):
        x = torch.from_numpy(x)
        x = torch.sigmoid(x)
        return x.numpy()


if __name__ == '__main__':
    explorer = Explorer(True)
    set1 = Set()
    for file_name in os.listdir('D:/data/object3/dataset'):
        s = time()
        image = cv2.imread(f'D:/data/object3/dataset/{file_name}')
        boxes = explorer.explore(image)
        print(boxes)
        for box, index in zip(boxes[0], boxes[1]):
            name = set1.category[index]
            x1 = int(box[1])
            y1 = int(box[2])
            x2 = int(box[3])
            y2 = int(box[4])
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            image = cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # image = cv2.resize(image,None,fx=2,fy=2)
        cv2.imshow('JK', image)
        if cv2.waitKey(0) == ord('c'):
            continue
