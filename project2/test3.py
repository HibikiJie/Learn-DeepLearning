from project2.zExplorerOpenCV import Explorer
from torchvision.transforms import ToTensor
from project2.P2Net import Net
from PIL import Image
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
to_tensor = ToTensor()
net.load_state_dict(torch.load('D:/data/object2/netParam/net13.pth'))
net = net.eval().to(device)


def iiii(p1,p2,p3):
    global yes
    global no
    global yes_mean
    global no_mean
    image1 = Image.open(p1)
    image2 = Image.open(p2)
    image3 = Image.open(p3)
    image1 = image1.convert('RGB')
    image2 = image2.convert('RGB')
    image3 = image3.convert('RGB')

    image1 = image1.resize((112,112),Image.ANTIALIAS)
    image1= to_tensor(image1).unsqueeze(0).to(device)
    image2 = image2.resize((112,112),Image.ANTIALIAS)
    image2= to_tensor(image2).unsqueeze(0).to(device)
    image3 = image3.resize((112,112),Image.ANTIALIAS)
    image3= to_tensor(image3).unsqueeze(0).to(device)

    f1 = net(image1)
    f2 = net(image2)
    f3 = net(image3)
    cos1 = torch.cosine_similarity(f1, f2).cpu().item()
    cos2 = torch.cosine_similarity(f1, f3).cpu().item()
    cos3 = torch.cosine_similarity(f2, f3).cpu().item()
    print(cos1)
    print(cos2)
    print(cos3)
    if cos1 >0.85 and cos2<0.85 and cos3<0.85:
        yes += 1
    else:
        no+=1
    yes_mean += cos1
    no_mean += cos2
    no_mean += cos3


path = 'D:/data/object2/test'
yes = 0
no = 0
yes_mean = 0
no_mean = 0
对比次数 = 0

for i in range(7):

    for j in range(7):
        if i==j:
            continue
        p1 = None
        count = 0
        for x1 in os.listdir(f'{path}/{i}'):
            p2 = p1
            p1 = f'{path}/{i}/{x1}'
            if count == 0:
                count += 1
                continue
            # count += 1
            # if count >=4:
            #     count=0
            #     break
            for x3 in os.listdir(f'{path}/{j}'):

                p3 = f'{path}/{j}/{x3}'
                # 对比次数+=1
                # if 对比次数>=2:
                #     对比次数=0
                #     break
                print('====%d=======%d======' % (i, j))
                iiii(p1, p2, p3)
print('===最后结果+++')
print(yes)
print(no)
print(yes/(yes+no))
print(yes_mean/(yes+no))
print(no_mean/(2*yes+2*no))
print(yes+no)