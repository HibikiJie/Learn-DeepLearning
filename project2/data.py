from torch.utils.data import Dataset
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
                one_hot = torch.zeros(3)
                one_hot[int(taget)-1] = 1
                one_hot = one_hot.type(torch.FloatTensor)
                self.dataset.append((img_path,one_hot))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img_data = torch.load(data[0])
        taget = data[1]
        return img_data,taget

sun_class = SunDataset()

print(sun_class[0])
print(sun_class[10000])