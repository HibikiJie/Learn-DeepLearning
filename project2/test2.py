import cv2
import os

path1 = 'D:/data/object2/CASIAWebFaceData/CASIAWebFaceData'
save_path = 'D:/data/object2/faceimage'
i = 0
j = 0
for files in os.listdir(path1):
    if not os.path.exists(f'{save_path}/{i}'):
        os.makedirs(f'{save_path}/{i}')
    j = 0
    for image in os.listdir(f'{path1}/{files}'):
        image_path = f'{path1}/{files}/{image}'
        img = cv2.imread(image_path)
        cv2.imwrite(f'{save_path}/{i}/{j}.jpg', img)
        j += 1
    i += 1
    if i == 6000:
        break
