from PIL import Image,ImageDraw
import os
import numpy
import time
# image_path = 'D:/data/object1/iamge'
image_path = 'D:/data/object1/train/48/positive'
bbox_path = 'D:/data/object1/train/48/positive.txt'

image_files = os.listdir(image_path)
with open(bbox_path,'r') as file:
    txt = file.read().split()

datas = []
for i in range(len(txt)//5):
    datas.append(txt[5*i:5*i+5])

for data in datas[10000:]:

    x1,y1,x2,y2 =float(data[1])*48,float(data[2])*48,float(data[3])*48,float(data[4])*48
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2),

    iamge = Image.open(f'{image_path}/{data[0]}')
    draw = ImageDraw.Draw(iamge)
    draw.rectangle((x1,y1,x2+48,y2+48),outline='red',width=3)
    iamge.show()
    time.sleep(1)