from PIL import Image
import os
import random

def change_image(path,yellow):
    global i
    image_bg = Image.open(path)
    image_ft = Image.open(f'D:/data/chapter2/yellow/{yellow}.png')
    image_bg = image_bg.resize((300,300),resample=Image.ANTIALIAS)
    w = random.randint(60,150)
    x1 = random.randint(0,300-w)
    y1 = random.randint(0,300-w)
    image_ft=image_ft.resize((w,w),resample=Image.ANTIALIAS)
    r,g,b,a = image_ft.split()
    image_bg.paste(image_ft,(x1,y1),mask=a)
    x2 = x1+w
    y2 = y1+w
    file_name = f"{i}.{x1}.{y1}.{x2}.{y2}.jpg"
    image_bg = image_bg.convert('RGB')
    image_bg.save(f'D:/data/chapter2/shuju/{file_name}')

root1 = 'D:/data/chapter2/image'
i=1
for file_name in os.listdir(root1):
    path = f'{root1}/{file_name}'
    yellow = random.randint(1,20)
    try:
        change_image(path,yellow)
    except:
        print(1)
    i+=1

