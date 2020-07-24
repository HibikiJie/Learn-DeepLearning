import numpy
import torch
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import cv2
import tqdm

def chang_data(path):
    hdu_list = fits.open(path)
    hdu_list.verify('silentfix')
    img_data = hdu_list[1].data
    img_data = cv2.resize(img_data,(224,224))
    maxx = img_data.max()
    minn = img_data.min()
    cha = maxx-minn
    img_data = (img_data-minn)/cha
    return img_data


root1 = "D:/sunspot/trainset/continuum"
root2 = "D:/sunspot/trainset/magnetogram"
i=1
print(len(os.listdir(f'{root1}/betax')))
print(len(os.listdir(f'{root2}/betax')))

for file_name1,file_name2 in zip(os.listdir(f'{root1}/betax'),os.listdir(f'{root2}/betax')):
    img1 = chang_data(f"{root1}/betax/{file_name1}")
    img2 = chang_data(f"{root2}/betax/{file_name2}")

    img = numpy.array((img1,img2))
    img = torch.tensor(img,dtype=torch.float32)
    file_name = str(i).rjust(4,"0")
    torch.save(img,f'D:/sunspot/train/3/{file_name}')
    print(i)
    i += 1
