from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from project3.P3Set import Set
import numpy
from math import log


def nms_with_category(boxes, categorys):
    """根据类别的不同，进行非极大值抑制"""
    picked_boxes = []
    picked_category = []
    for category_index in range(4):
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
            iou = calculate_iou(max_box, left_box)
            index = iou < 0
            box1 = left_box[index]
        if box1.shape[0]>0:
            picked_boxes.append(box1[0])
            picked_category.append(numpy.array([category_index]))

    return numpy.vstack(picked_boxes), numpy.stack(picked_category)

def calculate_iou(box1,box2):
    area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
    area2 = (box2[:, 3] - box2[:, 1]) * (box2[:, 4] - box2[:, 2])

    x1 = numpy.maximum(box1[1], box2[:, 1])
    y1 = numpy.maximum(box1[2], box2[:, 2])
    x2 = numpy.minimum(box1[3], box2[:, 3])
    y2 = numpy.minimum(box1[4], box2[:, 4])

    intersection_area = numpy.maximum(0, x2 - x1) * numpy.maximum(
        0, y2 - y1)
    return intersection_area / numpy.minimum(area1, area2)

x = numpy.random.rand(14,5)
category = numpy.random.choice(4, 14)
print(x)
print(category)
a,b =nms_with_category(x,category)
print(a.shape,b.shape)