from project2.zExplorerOpenCV import Explorer
from project2.P2Net import Net
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
import numpy


class FeaturesDataSet(Dataset):

    def __init__(self, root='D:/data/object2/datas'):
        super(FeaturesDataSet, self).__init__()
        self.data_set = []
        self.target = 0
        self.targets = {}
        for files_name in os.listdir(root):
            name = files_name
            for file_name in os.listdir(f'{root}/{files_name}'):
                path = f'{root}/{files_name}/{file_name}'
                feature_tensor = torch.load(path)
                self.data_set.append((self.target, feature_tensor))
            self.targets[f'{self.target}'] = name
            self.target += 1

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        target, feature = self.data_set[item]
        return feature, torch.tensor(target)


class IdentifyFace:

    def __init__(self, is_cuda=True):
        self.is_cuda = is_cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and is_cuda else "cpu")
        self.net = Net().to(self.device)
        self.net.load_state_dict(torch.load('D:/data/object2/netParam/net24.pth'))
        self.net = self.net.eval()
        self.explorer = Explorer(self.is_cuda)
        self.feature_data_set = FeaturesDataSet()
        self.feature_data_loader = DataLoader(self.feature_data_set, len(self.feature_data_set), False)

    def get_boxes(self, image_RGB):
        boxes = self.explorer.explore(image_RGB)
        return boxes

    def __call__(self, video=0, ):
        count = 0
        video_capture = cv2.VideoCapture(video)
        informations = ()
        boxes = []
        while True:
            success, image = video_capture.read()
            # image = cv2.resize(image, None, fx=0.7, fy=0.7)
            if count % 1 == 0:
                image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = self.get_boxes(image_RGB)
                boxes = self.adjust_boxes(boxes)
                informations = self.comparison_feature(image_RGB, boxes)
            for information in informations:
                name = information[0]
                x1, y1, x2, y2 = information[1], information[2], information[3], information[4]
                c = (information[5] - 0.8) / 0.2
                image = cv2.putText(image, name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                hist_x2 = x2 + 10
                hist_y2 = int(y2 - (y2 - y1) * c)
                image = cv2.rectangle(image, (x2, y2), (hist_x2, hist_y2), (0, 0, 255), -1)
            for box in boxes:
                x1, y1, x2, y2 = box
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('JK', image)
            # cv2.imwrite(f'D:\data\object2\image/{count}.jpg',image)
            count += 1
            if count == 1000:
                count = 0
            if cv2.waitKey(1) == ord('q'):
                break

    def adjust_boxes(self, boxes):
        box_new = []
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            w = x2 - x1
            h = y2 - y1
            c_x = x1 + w / 2 - w * 0.02
            c_y = y1 + h / 2
            sid_length = 0.33 * w
            c_x, c_y, sid_length = int(c_x), int(c_y), int(sid_length)
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            # x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            box_new.append((x1, y1, x2, y2))
        return box_new

    def comparison_feature(self, image_RGB, boxes):
        information = []
        targets = self.feature_data_set.targets
        # print(targets)
        img_h, img_w, c = image_RGB.shape
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 > 0 and x2 < img_w and y1 > 0 and y2 < img_h and x2 - x1 >= 150:
                image_crop = image_RGB[y1:y2, x1:x2]
                image_crop = cv2.resize(image_crop, (112, 112), interpolation=cv2.INTER_AREA)
                image_tensor = torch.from_numpy(
                    numpy.transpose(image_crop, (2, 0, 1)) / 255).float().unsqueeze(0).to(self.device)
                feature = self.net(image_tensor).cpu()
                for features, target in self.feature_data_loader:
                    cos_thetas = torch.cosine_similarity(feature, features, dim=1)
                    max_index = torch.argmax(cos_thetas)

                    cos_theta = cos_thetas[max_index]
                    print(cos_theta.item())
                    if cos_theta >= 0.85:
                        name = targets[str(target[max_index].item())]
                        # print(name,cos_theta.item())
                        information.append((name, x1, y1, x2, y2, cos_theta.item()))
        return information


if __name__ == '__main__':
    indetify_face = IdentifyFace(True)

    indetify_face(video='D:/data/object2/has.mp4')
