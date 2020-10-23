import cv2
import onnxruntime
from project1.MTCNNCUDA.NMS import non_maximum_suppression
from time import time
import torch
import numpy
import cv2

# def to_tensor(image_numpy):
#     return torch.from_numpy(numpy.transpose(image_numpy, (2, 0, 1)) / 255 - 0.5).float()
#
#
# def to_numpy(x):
#     return x.numpy()


# img = cv2.imread('1.jpg')
# img = to_tensor(img)
# img.unsqueeze_(0)
#
# ort_session = onnxruntime.InferenceSession("mtcnn_onnx/p_net.onnx")
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
# ort_outs = ort_session.run(None, ort_inputs)

class Explorer:

    def __init__(self, is_cuda=False):
        '''设置网络参数'''
        self.p_confidence_threshold = 0.9  # 建议值0.6
        self.p_nms_threshold = 0.4  # 建议值0.6
        self.r_confidence_threshold = 0.99  # 建议值0.6
        self.r_nms_threshold = 0.3  # 建议值0.5
        self.o_confidence_threshold = 0.9999  # 建议值0.9999
        self.o_nms_threshold = 0.3  # 建议值0.7

        '''实例化网络'''
        self.p_net = onnxruntime.InferenceSession("mtcnn_onnx/p_net.onnx")
        self.r_net = onnxruntime.InferenceSession("mtcnn_onnx/r_net.onnx")
        self.o_net = onnxruntime.InferenceSession("mtcnn_onnx/o_net.onnx")

    def explore(self, image):
        start_time = time()
        boxes_p = self.p_net_explore(image)
        if not boxes_p.size > 0:
            return numpy.array([])
        boxes_r = self.r_net_explore(image, boxes_p)
        if not boxes_r.size > 0:
            return numpy.array([])
        boxes_o = self.o_net_explore(image, boxes_r)
        print(time() - start_time)
        return boxes_o

    def p_net_explore(self, image):
        image = image
        boxes_p = []
        confidences = []
        h, w, c = image.shape
        min_side_len = min(w, h)
        zoom_ratio = 1
        while min_side_len > 12:
            image_tensor = self.to_tensor(image).unsqueeze(0)
            inputs = {'input': self.to_numpy(image_tensor)}
            out = self.p_net.run(None, inputs)
            confidence, position = out
            confidence = confidence[0][0]
            position = position[0]

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
            image = cv2.resize(image, (_w, _h))
            min_side_len = min(_h, _w)
        boxes_p = numpy.vstack(boxes_p)
        confidences = numpy.hstack(confidences)
        return non_maximum_suppression(boxes_p, confidences, self.p_nms_threshold)

    def r_net_explore(self, image, boxes):
        image_tensors, xys, side_lens = self.cut_out_image(image, boxes, 24)
        inputs = {'input': self.to_numpy(image_tensors)}
        out = self.r_net.run(None, inputs)
        confidence, position = out
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
        inputs = {'input': self.to_numpy(image_tensors)}
        out = self.o_net.run(None, inputs)
        confidence, positon = out
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
        __h, __w, c = image.shape
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w // 2
        center_y = y1 + h // 2
        side_lens = numpy.maximum(w, h)

        x1 = numpy.maximum(center_x - side_lens * 0.5, 0).astype('int')
        y1 = numpy.maximum(center_y - side_lens * 0.5, 0).astype('int')
        x2 = numpy.minimum(x1 + side_lens, __w)
        y2 = numpy.minimum(y1 + side_lens, __h)

        side_lens = numpy.minimum(y2 - y1, x2 - x1).astype('int')
        x2 = x1 + side_lens
        y2 = y1 + side_lens

        xys = numpy.stack((x1, y1, x2, y2), 1)
        images = []
        # print(xys)
        for xy in xys:
            __x1 = int(xy[0])
            __y1 = int(xy[1])
            __x2 = int(xy[2])
            __y2 = int(xy[3])
            image_crop = image[__y1:__y2, __x1:__x2]
            image_resize = cv2.resize(image_crop, (size, size), interpolation=cv2.INTER_AREA)
            image_tensor = self.to_tensor(image_resize)
            images.append(image_tensor)
        return torch.stack(images), xys, side_lens

    def to_tensor(self, image_numpy):
        return torch.from_numpy(numpy.transpose(image_numpy, (2, 0, 1)) / 255 - 0.5).float()

    def to_numpy(self,x):
        return x.numpy()


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    explorer = Explorer(True)
    i = 0
    boxes = None
    a_time = time()
    frames = []
    n = 608
    while True:
        success, img = video_capture.read()
        # img = cv2.resize(img,(480,480))
        if success and (i % 2 == 0):
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
            sid_length = max(0.4 * w, 0.3 * h) * 0.82
            c_x, c_y, sid_length = int(c_x), int(c_y), int(sid_length)
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('JK', img)
        i += 1
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()