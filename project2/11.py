import tqdm
import time
import torch
import numpy
import matplotlib.pyplot as plt
from astropy.io import fits
# for i in range(10):
#     for i in tqdm.tqdm(range(20)):
#         time.sleep(0.5)
# y = torch.tensor([[0.1,0.9,0.1],
#                   [0.1,0.8,0.1]],dtype=torch.float32)
# taget = torch.tensor([[0,1,0],[0,1,0]],dtype=torch.float32)
# weight = torch.tensor([1,2,1])
# f=torch.nn.BCELoss()
# loss = f(y,taget)
# print(loss)
# a = torch.tensor([[0,0,1]],dtype=torch.float32)
# b = torch.tensor([[1,0,0]],dtype=torch.float32)
# pre_tage = torch.argmax(a,1)
# taget_tage = torch.argmax(b,1)
# c = torch.eq(pre_tage,taget_tage)
# print(pre_tage.item())
# # print(taget_tage)
# epoch=1
# with open("train.txt", 'w') as f:
#     f.write(str(epoch))
# a = numpy.array([numpy.nan,1,2])
# print(numpy.isnan(a).sum())
# # print(a)
for i in range(1,10000):
    file_name = str(i).rjust(4,"0")
    a = torch.load(f'D:/sunspot/test/3/{file_name}')
    c = numpy.isnan(a).sum()
    if c >0:
        print(i,c)
# plt.imshow(a[1])

# a = torch.load(f'D:/sunspot/train/1/1992')
# a = a.numpy()
# print(numpy.isnan(a).sum())
# print(a)

# hdu_list = fits.open('D:/sunspot/trainset/continuum/magnetogram/hmi.sharp_720s.3240.20131001_044800_TAI.continuum.fits')
# hdu_list.verify('silentfix')
# img_data = hdu_list[1].data
# plt.imshow(img_data)
# plt.show()