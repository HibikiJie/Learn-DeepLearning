from project2.zExplorerOpenCV import Explorer
from project2.P2Net import Net
import cv2
import torch
import numpy
import os
from torchvision.models import densenet121
# video = 'C:/Users/lieweiai/Pictures/737062022a9aa1679c9d865c43c413e4.mp4'
# video = 'http://admin:admin@192.168.42.129:8081/video'
# video = "http://admin:admin@192.168.0.121:8081/video"
video = 'D:/data/object2/has.mp4'
video_capture = cv2.VideoCapture(video)
explorer = Explorer(True)
boxes = None
i = 0
n = 0
image = None
net = Net()
# net = densenet121()
net.load_state_dict(torch.load('D:/data/object2/netParam/net24.pth'))
net = net.eval()
name = 'DQCX'
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
            c_y = y1 + h / 2
            sid_length = 0.33 * w
            c_x, c_y, sid_length = int(c_x), int(c_y), int(sid_length)
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if i % 8 == 0:
                if x1 > 0 and x2 < img_w and y1 > 0 and y2 < img_h:
                    if x2-x1>150:
                        image_crop = image[y1:y2, x1:x2]
                        image_crop = cv2.resize(image_crop, (112, 112), interpolation=cv2.INTER_AREA)
                        image_tensor = torch.from_numpy(
                            numpy.transpose(image_crop, (2, 0, 1)) / 255).float().unsqueeze(0)
                        feature = net(image_tensor).squeeze(0)
                        if not os.path.exists(f'D:/data/object2/datas/{name}'):
                            os.makedirs(f'D:/data/object2/datas/{name}')
                        torch.save(feature, f'D:/data/object2/datas/{name}/{n}.ft')
                        # cv2.imwrite(f'D:/data/object2/datas/{name}/{n}.jpg',image_crop)
                        n+=1
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('JK', img)
        i += 1
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()