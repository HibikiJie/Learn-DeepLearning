import torch
import time
import numpy
'''
1. 设定目标框的置信度阈值
2. 根据置信度降序排列候选框列表
3. 选取置信度最高的框A添加到输出列表，并将其从候选框列表中删除
4. 计算A与候选框列表中的所有框的IoU值，删除大于阈值的候选框
5. 重复上述过程，直到候选框列表为空，返回输出列表
'''


def compute_iou(box1, box2, is_min=False):
    """
    计算两候选框之间的交并比
    :param box1: 第一个候选框
    :param box2: 第二个候选框
    :return: 两候选框之间的交并比
    """
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    x1 = numpy.maximum(box1[0], box2[:, 0])
    y1 = numpy.maximum(box1[1], box2[:, 1])
    x2 = numpy.minimum(box1[2], box2[:, 2])
    y2 = numpy.minimum(box1[3], box2[:, 3])

    intersection_area = numpy.maximum(0, x2 - x1) * numpy.maximum(
        0, y2 - y1)

    if is_min:
        return intersection_area / numpy.minimum(area1, area2)
    else:
        return intersection_area / (area1 + area2 - intersection_area)


def non_maximum_suppression(boxes_place, boxes_confidence, threshold, is_min=False):
    if boxes_confidence.shape[0] == 0:
        return numpy.array([])
    # start_time = time.time()
    '''设定存放输出的数据'''
    picked_boxes = []
    '''获取置信度排序，按照降序排列'''
    order = numpy.argsort(boxes_confidence)[::-1]
    boxes_place = boxes_place[order]

    while boxes_place.shape[0] > 1:
        '''  选出当前置信度最大的索引'''
        max_box = boxes_place[0]
        picked_boxes.append(max_box)
        leftover_boxes = boxes_place[1:]
        '''将当前最大置信度的候选框，与剩下的候选框计算交并比'''

        iou = compute_iou(max_box, leftover_boxes, is_min)

        '''保留下小于阈值的候选框的位置索引'''
        index = numpy.where(iou < threshold)
        boxes_place = leftover_boxes[index]
    if boxes_place.shape[0]>0:
        picked_boxes.append(boxes_place[0])
    # print('NMS_COST_TIME:',time.time()-start_time)
    return numpy.stack(picked_boxes)


if __name__ == '__main__':
    coor = torch.tensor([[59, 120, 137, 368],
                         [221, 89, 369, 367],
                         [54, 154, 148, 382]],dtype=torch.float32).numpy()

    conf = torch.tensor([0.124648176,
                         0.35818103,
                         0.13638769]).numpy()
    print(conf.shape)
    start = time.time()
    for i in range(1000):
        x = non_maximum_suppression(coor, conf, 0.6, False)

    print((time.time() - start)/1000)
