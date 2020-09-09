import cv2

image = cv2.imread('C:/Users/lieweiai/Desktop/Chapter05.png')
print(image.shape)
img1 = image[0:10000]
img2 = image[10000:]
cv2.imwrite('C:/Users/lieweiai/Desktop/1.png',img1)
cv2.imwrite('C:/Users/lieweiai/Desktop/2.png',img2)