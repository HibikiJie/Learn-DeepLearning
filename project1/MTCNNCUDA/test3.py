import cv2
from project1.MTCNNCUDA.Explorer import Explorer
import os


def image_catch(image_files_path,save_path):
    image_files_path = image_files_path
    explorer = Explorer(True)
    i = 600
    for file in os.listdir(image_files_path):
        image_path = f'{image_files_path}/{file}'
        img = cv2.imread(image_path)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = explorer.explore(image)
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            w = x2 - x1
            h = y2 - y1
            c_x = x1 + w / 2 - w * 0.02
            c_y = y1 + h / 2 + h * 0.025
            sid_length = max(0.4 * w, 0.3 * h) * 0.95
            c_x, c_y, sid_length = int(c_x), int(c_y), int(sid_length)
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            image_crop = img[y1:y2, x1:x2]
            cv2.imwrite(f'{save_path}/{i}.jpg', image_crop)
            i+=1
            break

image_catch('D:/data/object2/feiliaoaaaa','D:/data/object2/feiliao')