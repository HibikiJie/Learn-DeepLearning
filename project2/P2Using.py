from project2.zExplorerOpenCV import Explorer
from project2.P2Net import Net
import cv2
import torch
import numpy
import os
from torchvision.models import densenet121
video = 'C:/Users/lieweiai/Pictures/737062022a9aa1679c9d865c43c413e4.mp4'
# video = 'http://admin:admin@192.168.42.129:8081/video'
# video = "http://admin:admin@192.168.0.121:8081/video"
# video = 'D:/data/object2/qiaoben.mp4'
video_capture = cv2.VideoCapture(0)
explorer = Explorer(True)
boxes = None
i = 0
image = None
net = Net()
# net = densenet121()
# net.load_state_dict(torch.load('D:/data/object2/netParam/net233.pth'))
features = []
net = net.eval()
for feature in os.listdir('D:/data/object2/datas'):
    feature_data = torch.load(f'D:/data/object2/datas/{feature}').squeeze(0)
    features.append(feature_data)
features = torch.stack(features)
features = torch.nn.functional.normalize(features,dim=1)
# exit()
is_draw = False
while True:
    success, img = video_capture.read()
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # print(img.shape)
    if success:
        img_h, img_w, c = img.shape
        if i % 2 == 0:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = explorer.explore(image)
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            w = x2 - x1
            h = y2 - y1
            c_x = x1 + w / 2 - w * 0.02
            c_y = y1 + h / 2 + h * 0.025
            sid_length = max(0.4 * w, 0.3 * h) * 0.95
            c_x, c_y, sid_length = int(c_x), int(c_y), int(sid_length)
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if i % 8 == 0:
                if x1 > 0 and x2 < img_w and y1 > 0 and y2 < img_h and x2-x1>112:
                    image_crop = image[y1:y2, x1:x2]
                    image_crop = cv2.resize(image_crop, (128, 128), interpolation=cv2.INTER_AREA)
                    image_tensor = torch.from_numpy(
                        numpy.transpose(image_crop, (2, 0, 1)) / 255 - 0.5).float().unsqueeze(0)
                    feature = net(image_tensor)
                    print(feature.max())
                    print(feature.min())
                    feature = torch.nn.functional.normalize(feature,dim=1).squeeze(0)
                    cos_theta = torch.matmul(features,feature)
                    # cos_theta = torch.sum(cos_theta,dim=1,keepdim=True)
                    print(cos_theta)
                    c = cos_theta.max()
                    print(c)
                    if c > 0.9:
                        is_draw = True
                    else:
                        is_draw = False
            if is_draw:
                cv2.putText(img, "Xiangjie", (x1, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('JK', img)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()
