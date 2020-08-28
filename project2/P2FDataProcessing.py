import cv2
from project1.MTCNNCUDA.Explorer import Explorer
import os
from PIL import Image

def video_catch():
    # video = 'C:/Users/lieweiai/Pictures/737062022a9aa1679c9d865c43c413e4.mp4'
    video = 'http://admin:admin@192.168.42.129:8081/video'
    # video = "http://admin:admin@192.168.0.121:8081/video"
    video_capture = cv2.VideoCapture(video)
    explorer = Explorer(True)
    i = 0
    boxes = None

    # out = cv2.VideoWriter('2.avi')
    frames = []
    while True:
        success, img = video_capture.read()
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        print(img.shape)
        if success and (i % 5 == 0):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = explorer.explore(image)
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            w = x2 - x1
            h = y2 - y1
            c_x = x1 + w / 2
            c_y = y1 + h / 2
            c_x = int(c_x)
            c_y = int(c_y)
            sid_length = int(max(0.4 * w, 0.3 * h))
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            image_crop = img[y1:y2, x1:x2]
            cv2.imwrite(f'D:/data/object2/2/{i}.jpg', image_crop)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('JK', img)
        i += 1
        if cv2.waitKey(1) == ord('q'):
            break


def image_catch(image_files_path,save_path):
    image_files_path = image_files_path
    explorer = Explorer(True)
    i = 0
    for file in os.listdir(image_files_path):
        image_path = f'{image_files_path}/{file}'
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = explorer.explore(image)
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            w = x2 - x1
            h = y2 - y1
            c_x = x1 + w / 2
            c_y = y1 + h / 2
            c_x = int(c_x)
            c_y = int(c_y)
            sid_length = int(max(0.4 * w, 0.3 * h))
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            image_crop = img[y1:y2, x1:x2]
            cv2.imwrite(f'{save_path}/{i}.jpg', image_crop)
            i+=1
            break


if __name__ == '__main__':
    # for i in range(6):
    #     j = 0
    #     for file in os.listdir(f'D:/data/object2/{i}'):
    #         image = Image.open(f'D:/data/object2/{i}/{file}')
    #         w,h = image.size
    #         if w > 150 and h>150:
    #             image = image.resize((128, 128), Image.ANTIALIAS)
    #             image.save(f'D:/data/object2/data/{i}/{j}.jpg')
    #             j += 1

    for i in (9,10):
        j = 0
        for file in os.listdir(f'D:/data/object2/object2/{i}'):
            image = Image.open(f'D:/data/object2/object2/{i}/{file}')
            w, h = image.size
            if w > 100 and h > 100:
                image = image.resize((112, 112), Image.ANTIALIAS)
                image.save(f'D:/data/object2/temp/{i}/{j}.jpg')
                j += 1