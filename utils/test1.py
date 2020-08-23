import cv2
for i in range(17):
    image = cv2.imread(f'D:\data\chapter3\image/epoch{i}.png')
    image = cv2.resize(image,(1920,961))
    # cv2.imshow("jk",image)
    cv2.imwrite(f'D:\data\chapter3\image/epoch{i}.png',image)