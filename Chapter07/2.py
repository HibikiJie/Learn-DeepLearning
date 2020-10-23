import torch
import cv2

image = cv2.imread('xinchuan.jpg')
cv2.imwrite('xinchuan.jpg',image[0:360,0:360])