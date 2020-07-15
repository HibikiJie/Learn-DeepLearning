import numpy
import torch
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import cv2
root = 'D:/sunspot/trainset/magnetogram/betax'


def chang_data(path):
    hdu_list = fits.open(path)
    hdu_list.verify('silentfix')
    img_data = hdu_list[1].data
    img_data = cv2.resize(img_data,(224,224))
    max = img_data.max()
    img_data = img_data/max
    return img_data



root1 = "D:/sunspot/trainset/continuum"
root2 = "D:/sunspot/trainset/magnetogram"
i=1
print(len(os.listdir(f'{root1}/al')))
print(len(os.listdir(f'{root2}/betax')))

for file_name1,file_name2 in zip(os.listdir(f'{root1}/betax'),os.listdir(f'{root2}/betax')):
    img1 = chang_data(f"{root1}/betax/{file_name1}")
    img2 = chang_data(f"{root2}/betax/{file_name2}")
    img = numpy.array((img1,img2))
    img = torch.tensor(img)
    img = img.type(torch.FloatTensor)

    file_name = str(i).rjust(5,"0")
    torch.save(img,f'D:/sunspot/train/3/{file_name}')
    print(i)
    i += 1

import torch.utils.data.sampler