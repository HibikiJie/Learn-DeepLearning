from torch.utils.data import Dataset,WeightedRandomSampler,DataLoader
import torch
import os

class SunDataset(Dataset):

    def __init__(self,root="D:/sunspot",is_train=True):
        super(SunDataset, self).__init__()
        self.root = root
        self.dataset = []

        sub_dir = "train" if is_train else "test"
        for taget in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{taget}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                one_hot = torch.zeros(3,dtype=torch.float32)
                one_hot[int(taget)-1] = 1
                self.dataset.append((img_path,one_hot))
        self.weights = []
        for i in range(4007):
            self.weights.append(15)
        for i in range(5353):
            self.weights.append(12)
        for i in range(2107):
            self.weights.append(20)
        self.simpler = WeightedRandomSampler(self.weights,len(self.dataset),True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img_data = torch.load(data[0])
        taget = data[1]
        return img_data,taget


if __name__ == '__main__':

    sun_class = SunDataset()

    dataload = DataLoader(sun_class,60,sampler=sun_class.simpler)
    for i in dataload:
        print(i[1].sum(dim=0))
        exit()