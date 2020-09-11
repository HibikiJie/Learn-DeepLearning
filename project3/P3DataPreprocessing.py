from xml.etree import cElementTree as ET
import cv2
import os


def catch_information():
    """解析xml文件为txt"""
    path = 'D:/data/object3/VOC2012/Annotations'
    images_path = 'D:/data/object3/VOC2012/JPEGImages'
    targets = []
    for file_name in os.listdir(path):
        file = f'{path}/{file_name}'
        tree = ET.parse(file)
        name = tree.findtext('filename')
        image_path = f'{images_path}/{name}'
        image = cv2.imread(image_path)
        massage = name
        print(file)
        for obj in tree.iter('object'):
            x1 = obj.findtext('bndbox/xmin')
            y1 = obj.findtext('bndbox/ymin')
            x2 = obj.findtext('bndbox/xmax')
            y2 = obj.findtext('bndbox/ymax')
            x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            name = obj.findtext('name')
            if name not in targets:
                targets.append(name)
            massage = f'{massage}  {name}  {x1}  {y1}  {x2}  {y2}'
            for part in obj.iter('part'):
                x1 = part.findtext('bndbox/xmin')
                y1 = part.findtext('bndbox/ymin')
                x2 = part.findtext('bndbox/xmax')
                y2 = part.findtext('bndbox/ymax')
                name = part.findtext('name')
                x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
                # image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                massage = f'{massage}  {name}  {x1}  {y1}  {x2}  {y2}'
                if name not in targets:
                    targets.append(name)
        print(massage)
        # with open('D:/data/object3/voc2012.txt', 'a') as f:
        #     f.write(massage + '\n')
        cv2.imshow('JK', image)
        if cv2.waitKey() == ord('c'):
            continue


catch_information()
