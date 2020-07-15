import numpy
import os
import cv2

'''================================================='''
train_data_path = 'D:/sun/train_jpg_input'
test_data_path = 'D:/sun/test_jpg_input'

'''================================================='''
def chang_image(path):
    img = cv2.imread("D:/sun/test_jpg_input/"+path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_back = numpy.zeros((224,224),numpy.uint8)
    img_back[:,:]=128
    h,w = img_gray.shape
    print(h,w)
    if h>=w and h>224:
        ratio = 224/h
        img_gray = cv2.resize(img_gray,None,fx=ratio,fy=ratio)
        h, w = img_gray.shape
    elif w>h and w>224:
        ratio = 224/w
        img_gray = cv2.resize(img_gray,None,fx=ratio,fy=ratio)
        h,w = img_gray.shape
        print('yes')
    w = w-w%2
    h = h - h%2
    img_gray = cv2.resize(img_gray,(w,h))
    print(img_gray.shape)
    w = w//2
    h = h//2
    img_back[112-h:112+h,112-w:112+w] = img_gray
    cv2.imshow('jk', img_back)
    cv2.imshow('jk2', img_gray)
    cv2.waitKey(1)
    return img_back

list_dir = os.listdir(test_data_path)

print(list_dir)

for path in list_dir:
    try:
        img = chang_image(path)

        cv2.imwrite('D:/sun/test/'+path,img)
    except:
        pass



