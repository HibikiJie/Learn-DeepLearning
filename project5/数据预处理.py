import cv2
import os
def a1():
    images_path = 'D:/data/merge_no_smoke/images'
    labels_path = 'D:/data/merge_no_smoke/labels'
    for label_path in os.listdir(labels_path):

        image_path = label_path[:-4] + '.jpg'
        image = cv2.imread(f'{images_path}/{image_path}')
        label = []
        with open(f'{labels_path}/{label_path}') as file:
            for line in file.readlines():
                label.append(line.strip().split())
        h,w,c = image.shape
        massage = f'{images_path}/{image_path}'
        for box in label:
            cl = box[0]
            c_x = float(box[1])*w
            c_y = float(box[2])*h
            _w = float(box[3])*w
            _h = float(box[4])*h
            x1 = c_x-_w/2
            x2 = c_x+_w/2
            y1 = c_y - _h/2
            y2 = c_y + _h/2
            # print(x1,y1)
            image = cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            massage += f'  {cl}  {int(x1)}  {int(y1)}  {int(x2)}  {int(y2)}'
        cv2.imshow('jk',image)
        cv2.waitKey(1)
        with open('labels.txt','a') as file:
            file.write(massage+'\n')
