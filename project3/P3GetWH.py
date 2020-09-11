import numpy

with open('D:/data/object3/train.txt') as file:
    for line in file.readlines():
        line = line.split()
        boxes = line[1:]
        for i in range(len(boxes)//5):
            box = boxes[5*i:5*i+5]
            x1 = int(box[1])
            y1 = int(box[2])
            x2 = int(box[3])
            y2 = int(box[4])
            w = x2-x1
            h = y2-y1
            print(w,h)
            with open('D:/data/object3/wh.txt','a') as f2:
                f2.write('{}  {}\n'.format(w, h))