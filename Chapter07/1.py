import cv2
image_name = '4.jpg'
image = cv2.imread(image_name)
h, w, _ = image.shape
max_len = min(h,w)
image = cv2.resize(image, None, fx=512/max_len,fy=512/max_len)
h, w, _ = image.shape
max_len = min(h,w)
print(image[h//2-256:h//2+256, w//2-256:w//2+256].shape)
cv2.imwrite(image_name, image[h//2-256:h//2+256, w//2-256:w//2+256])