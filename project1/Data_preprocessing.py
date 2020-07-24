import numpy
import os
import cv2
import torch
import tqdm
'''================================================='''
train_data_path = 'D:/sun/train_jpg_input'

'''================================================='''
def chang_image(path):
    file_path = f'D:/sun/train_jpg_input/{path}'
    img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(224,224))
    img_data = torch.tensor([img],dtype=torch.float32)
    img_data = img_data/255
    return img_data.unsqueeze(0)



file1 = open('D:/sun/train_output.txt')
file_data1 = file1.read().split()
file2 = open('D:/sun/train/train.txt')
file_data2 = file2.read().split()
a = len(file_data1)//2

para = None
target = None
for i in tqdm.tqdm(range(a)):
    path = file_data1[2 * i]
    img = chang_image(path)

    torch.save(img,f'D:/data/object1/train/{i}')
    if i == 0:
        para = numpy.array(file_data2[11*i+1:11*i+11],dtype=numpy.float32)
        para = torch.tensor([[para]],dtype=torch.float32)
        target = torch.tensor([[int(file_data1[2*i+1])]], dtype=torch.float32)
        continue
    para_temp = numpy.array(file_data2[11 * i + 1:11 * i + 11], dtype=numpy.float32)
    para_temp = torch.tensor([[para_temp]], dtype=torch.float32)
    target_temp = torch.tensor([[int(file_data1[2*i+1])]], dtype=torch.float32)
    para = torch.cat([para,para_temp],dim=0)
    target = torch.cat([target, target_temp], dim=0)
torch.save(target,'D:/data/object1/target')
torch.save(para,'D:/data/object1/para')




