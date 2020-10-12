from project3.P3YOLO_V3 import YOLOVision3Net
from project3.P3Set import Set
import torch
import numpy
import cv2
from project3.P3Explorer import Explorer


def main():
    vedio = cv2.VideoCapture(0)
    explorer = Explorer(True)
    set1 = Set()
    boxes = (None, None)
    a_w = 0
    a_h = 0
    while True:
        success, image = vedio.read()
        if success:
            h, w, c = image.shape
            a_h = h / 416
            a_w = w / 416
            boxes = explorer.explore(cv2.resize(image, (416, 416)))
        for box, index in zip(boxes[0], boxes[1]):
            name = set1.category[index]
            x1 = int(box[1] * a_w)
            y1 = int(box[2] * a_h)
            x2 = int(box[3] * a_w)
            y2 = int(box[4] * a_h)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            image = cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        image = cv2.resize(image, None, fx=2, fy=2)
        cv2.imshow('JK', image)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
