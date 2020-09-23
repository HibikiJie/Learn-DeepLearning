import cv2
import numpy

category = ['person', 'aeroplane', 'tvmonitor', 'train', 'boat', 'dog', 'chair', 'bird',
            'bicycle', 'bottle', 'sheep', 'diningtable', 'horse', 'motorbike', 'sofa', 'cow', 'car', 'cat', 'bus',
            'pottedplant']

save_path = 'D:/data/object3/dataset'
images_path = 'D:/data/object3/VOC2012/JPEGImages'
background = numpy.zeros((416, 416, 3))

with open('D:/data/object3/voc.txt') as f:
    for line in f.readlines():
        line = line.split()
        file_name = line[0]
        file_path = f'{images_path}/{file_name}'
        image = cv2.imread(file_path)
        h1, w1, c1 = image.shape
        max_len = max(h1, w1)
        fx = 416 / max_len
        fy = 416 / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, c2 = image.shape
        background = numpy.zeros((416, 416, 3), dtype=numpy.uint8)
        s_h = 208 - h2 // 2
        s_w = 208 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        print(background.shape)
        targets = line[1:]
        a = len(targets) // 5
        massage = file_name
        for i in range(a):
            box = targets[5 * i:5 * i + 5]
            if box[0] == 'head' or box[0] == 'hand' or box[0] == 'foot':
                continue
            target = category.index(box[0])
            x1 = int((int(box[1]) - w1 / 2) * fx + 208)
            y1 = int((int(box[2]) - h1 / 2) * fy + 208)
            x2 = int((int(box[3]) - w1 / 2) * fx + 208)
            y2 = int((int(box[4]) - h1 / 2) * fy + 208)
            massage = f'{massage}  {target}  {x1}  {y1}  {x2}  {y2}'
            background = cv2.rectangle(background,(x1,y1),(x2,y2),(0,255,255))
            cv2.putText(background,box[0],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        with open('D:/data/object3/TrainMobile.txt', 'a') as f2:
            massage += '\n'
            f2.write(massage)
        cv2.imshow('JK',background)
        cv2.waitKey(1)