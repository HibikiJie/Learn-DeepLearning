from torch import nn
from torch.utils.data import Dataset, DataLoader
from zipfile import ZipFile
import torch
import numpy
import cv2
from tqdm import tqdm


class VCDataSet(Dataset):

    def __init__(self, root, train=True):
        super(VCDataSet, self).__init__()
        self.codes = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'a': 10,
                      'B': 11, 'b': 11, 'C': 12, 'c': 12, 'D': 13, 'd': 13, 'E': 14, 'e': 14, 'F': 15, 'f': 15, 'G': 16,
                      'g': 16, 'H': 17, 'h': 17, 'I': 18, 'i': 18, 'J': 19, 'j': 19, 'K': 20, 'k': 20, 'L': 21, 'l': 21,
                      'M': 22, 'm': 22, 'N': 23, 'n': 23, 'O': 24, 'o': 24, 'P': 25, 'p': 25, 'Q': 26, 'q': 26, 'R': 27,
                      'r': 27, 'S': 28, 's': 28, 'T': 29, 't': 29, 'U': 30, 'u': 30, 'V': 31, 'v': 31, 'W': 32, 'w': 32,
                      'X': 33, 'x': 33, 'Y': 34, 'y': 34, 'Z': 35, 'z': 35}
        self.root = root
        if train:
            self.root += '/train.zip'
        else:
            self.root += '/test.zip'
        self.zip_files = ZipFile(self.root)
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                image_name = file_name.split('/')[-1]
                target = image_name.split('.')[0]
                self.data_set.append((file_name, (self.codes[target[0]], self.codes[target[1]], self.codes[target[2]],
                                                  self.codes[target[3]])))
        print('数据初始化完成')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        image_name, target = self.data_set[item]
        target = torch.tensor(target)
        image = self.zip_files.read(image_name)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = torch.from_numpy(image).float()
        image = image / 255
        return image, target


class GRUNet(nn.Module):

    def __init__(self):
        super(GRUNet, self).__init__()
        self.con_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.gru1 = nn.GRU(1536, 512, 3, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        self.classification = nn.Linear(256, 36)

    def forward(self, input_):
        input_ = input_.permute(0, 3, 1, 2)
        input_ = self.con_layer(input_)

        input_ = input_.permute(0, 3, 1, 2).reshape(-1, 28, 256*6)
        input_, h_n = self.gru1(input_)
        out_gru1 = input_[:, -1, :]
        out_gru1 = out_gru1.unsqueeze(1)
        out_gru1 = out_gru1.expand(-1, 4, 512)
        out_gru2, h_n2 = self.gru2(out_gru1)
        return self.classification(out_gru2.reshape(-1, 256))


def train():
    train_data_set = VCDataSet('D:/data/chapter6')
    train_data_loader = DataLoader(train_data_set, 256, True)
    net = GRUNet().cuda()
    net.load_state_dict(torch.load('D:/data/chapter6/net.pth'))
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = nn.CrossEntropyLoss().cuda()
    i = 0
    while True:
        loss_sum = 0
        for images, targets in tqdm(train_data_loader):
            images = images.cuda()
            targets = targets.reshape(-1).cuda()
            output = net(images)

            loss = loss_func(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
        i += 1
        torch.save(net.state_dict(),'D:/data/chapter6/net.pth')
        print(f'epoch:{i} ; loss:{loss_sum / len(train_data_loader)}')


def test():
    test_data_set = VCDataSet('D:/data/chapter6', False)
    test_data_loader = DataLoader(test_data_set, 20, False)
    net = GRUNet().cuda().eval()
    net.load_state_dict(torch.load('D:/data/chapter6/net.pth'))
    correct = 0
    i = 1
    for images, targets in tqdm(test_data_loader):
        images = images.cuda()
        output = net(images).cuda()
        output = output.reshape(-1, 4, 36).cpu()
        output = output.argmax(2)
        acc = output == targets
        acc = acc.all(1)
        acc = acc.sum().item()
        correct += acc
        i += 1
        if i % 100 == 0:
            print(output, targets)
    print(f'acc:{correct / len(test_data_set)}')


if __name__ == '__main__':
    # train()
    test()