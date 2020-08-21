from project1.MTCNNCUDA.Net import PNet, RNet, ONet
from project1.MTCNNCUDA.NMS import non_maximum_suppression
from time import time
import torch
from torchvision.transforms import ToTensor
import numpy
from PIL import Image
import cv2


class Explorer:

    def __init__(self, is_cuda=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and is_cuda else "cpu")
        print(self.device)
        '''设置网络参数'''
        self.p_confidence_threshold = 0.99  # 建议值0.6
        self.p_nms_threshold = 0.5  # 建议值0.6
        self.r_confidence_threshold = 0.999  # 建议值0.6
        self.r_nms_threshold = 0.4  # 建议值0.5
        self.o_confidence_threshold = 0.9999  # 建议值0.9999
        self.o_nms_threshold = 0.3  # 建议值0.7

        '''实例化网络'''
        self.p_net = PNet()
        self.r_net = RNet()
        self.o_net = ONet()
        self.p_net = self.p_net.to(self.device)
        self.r_net = self.r_net.to(self.device)
        self.o_net = self.o_net.to(self.device)

        '''加载网络参数'''
        self.p_net.load_parameters()
        self.r_net.load_parameters()
        self.o_net.load_parameters()

        self.to_tensor = ToTensor()

    def explore(self, image_numpy):
        start_time = time()
        image = Image.fromarray(image_numpy)
        boxes_p = self.p_net_explore(image)
        print('P:', time() - start_time)
        # return boxes_p
        if not boxes_p.size > 0:
            return numpy.array([])
        boxes_r = self.r_net_explore(image, boxes_p)
        if not boxes_r.size > 0:
            return numpy.array([])
        boxes_o = self.o_net_explore(image, boxes_r)
        print(time() - start_time)
        return boxes_o

    def p_net_explore(self, image):
        start_time = time()
        boxes_p = []
        confidences = []
        w, h = image.size
        min_side_len = min(w, h)
        scale = 1
        zoom_ratio = 1
        while min_side_len > 12:
            image_tensor = self.to_tensor(image).unsqueeze(0) - 0.5
            image_tensor = image_tensor.to(self.device)
            confidence, position = self.p_net(image_tensor)
            confidence = confidence[0][0].cpu().data.numpy()
            position = position[0].cpu().data.numpy()

            position = numpy.transpose(position, (1, 2, 0))

            mask = numpy.where(confidence > self.p_confidence_threshold)

            confidence = confidence[mask]
            position = position[mask]

            h_index = mask[0]
            w_index = mask[1]

            '''反算候选框对应于原图的位置'''
            x1 = w_index * 2 / zoom_ratio
            y1 = h_index * 2 / zoom_ratio
            x2 = (w_index * 2 + 12) / zoom_ratio
            y2 = (h_index * 2 + 12) / zoom_ratio

            xy = numpy.stack((x1, y1, x2, y2), 1)

            side_len = 12 / zoom_ratio
            position = xy + position * side_len

            boxes_p.append(position)
            confidences.append(confidence)

            '''变换图形尺寸'''
            zoom_ratio *= 0.7
            _w = int(w * zoom_ratio)
            _h = int(h * zoom_ratio)
            image = image.resize((_w, _h), Image.ANTIALIAS)
            min_side_len = min(_w, _h)
        boxes_p = numpy.vstack(boxes_p)
        confidences = numpy.hstack(confidences)
        print('P网络监测时间：', time() - start_time)
        return non_maximum_suppression(boxes_p, confidences, self.p_nms_threshold)

    def r_net_explore(self, image, boxes):
        image_tensors, xys, side_lens = self.cut_out_image(image, boxes, 24)

        image_tensors = image_tensors.to(self.device)
        confidence, position = self.r_net(image_tensors)
        confidence = confidence.cpu().data.numpy()
        position = position.cpu().data.numpy()
        '''反算'''
        index = numpy.where(confidence > self.r_confidence_threshold)  # 取出大于置信度阈值的索引
        confidence = confidence[index]
        position = position[index[0]]
        boxes_origin = xys[index[0]]
        side_lens = side_lens[index[0]].reshape(-1, 1)
        boxes_r = boxes_origin + side_lens * position
        return non_maximum_suppression(boxes_r, confidence, self.r_nms_threshold)

    def o_net_explore(self, image, boxes):

        image_tensors, xys, side_lens = self.cut_out_image(image, boxes, 48)
        image_tensors = image_tensors.to(self.device)
        confidence, positon = self.o_net(image_tensors)
        confidence = confidence.cpu().data.numpy()
        positon = positon.cpu().data.numpy()
        '''反算'''
        index = numpy.where(confidence > self.o_confidence_threshold)  # 取出大于置信度阈值的索引

        confidence = confidence[index]
        boxes_origin = xys[index[0]]
        side_lens = side_lens[index[0]].reshape(-1, 1)
        positon = positon[index[0]]
        boxes_o = boxes_origin + side_lens * positon

        return non_maximum_suppression(boxes_o, confidence, self.o_nms_threshold, True)

    def cut_out_image(self, image, boxes, size):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w // 2
        center_y = y1 + h // 2
        side_lens = numpy.maximum(w, h)
        x1 = center_x - side_lens * 0.5
        y1 = center_y - side_lens * 0.5
        x2 = x1 + side_lens
        y2 = y1 + side_lens
        xys = numpy.stack((x1, y1, x2, y2), 1)
        images = []
        for xy in xys:
            xy = tuple(xy)
            image_crop = image.crop(xy)
            image_resize = image_crop.resize((size, size), Image.ANTIALIAS)
            image_tensor = self.to_tensor(image_resize) - 0.5
            images.append(image_tensor)
        return torch.stack(images), xys, side_lens


if __name__ == '__main__':
    video = 'C:/Users/lieweiai/Pictures/737062022a9aa1679c9d865c43c413e4.mp4'
    # video = 'http://admin:admin@192.168.42.129:8081/video'
    # video = "http://admin:admin@192.168.0.121:8081/video"
    video_capture = cv2.VideoCapture(video)
    explorer = Explorer(True)
    i = 0
    boxes = None
    while True:
        success, img = video_capture.read()

        if success and (i % 2 == 0):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes = explorer.explore(image)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # img = cv2.resize(img, None, fx=2, fy=2)
        cv2.imshow('JK', img)
        i += 1
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
