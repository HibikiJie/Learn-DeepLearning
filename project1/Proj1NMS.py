import torch

'''
1. 设定目标框的置信度阈值
2. 根据置信度降序排列候选框列表
3. 选取置信度最高的框A添加到输出列表，并将其从候选框列表中删除
4. 计算A与候选框列表中的所有框的IoU值，删除大于阈值的候选框
5. 重复上述过程，直到候选框列表为空，返回输出列表
'''
import time

def compute_iou(box1, box2, is_min=False):
    """
    计算两候选框之间的交并比
    :param box1: 第一个候选框
    :param box2: 第二个候选框
    :return: 两候选框之间的交并比
    """
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    x1 = torch.max(box1[0], box2[:, 0])
    y1 = torch.max(box1[1], box2[:, 1])
    x2 = torch.min(box1[2], box2[:, 2])
    y2 = torch.min(box1[3], box2[:, 3])

    intersection_area = torch.max(torch.tensor(0, dtype=torch.float32), x2 - x1) * torch.max(
        torch.tensor(0, dtype=torch.float32), y2 - y1)
    if is_min:
        return intersection_area / torch.min(area1, area2)
    else:
        return intersection_area / (area1 + area2 - intersection_area)


def non_maximum_suppression(predict_dict, threshold, is_min=False):
    """
    非极大值抑制
    :param predict_dict: 输入的候选框
    :param threshold: 交并比的阈值
    :return: 非极大值抑制之后的候选框们
    """
    print(predict_dict.shape)
    start = time.time()
    if predict_dict.shape[0] == 1:
        return torch.tensor([[]])
    '''阈值设定'''
    threshold = threshold

    '''获取得分'''
    score = predict_dict[:, 4]

    '''设定存放输出的数据的索引'''
    picked_boxes = []

    '''获取置信度排序，按照降序排列'''
    order = torch.argsort(score, descending=True)

    while order.size()[0] > 0:
        '''选出当前置信度最大的索引'''
        picked_boxes.append(order[0])

        '''将当前最大置信度的候选框，与剩下的候选框计算交并比'''
        iou = compute_iou(predict_dict[order[0]], predict_dict[order[1:]], is_min)

        '''保留下小于阈值的候选框的位置索引'''
        order = order[torch.where(iou < threshold)[0] + 1]
    end = time.time()
    print('NMS_COST_TIME:',end-start)
    return predict_dict[[picked_boxes]]


if __name__ == '__main__':
    predict_dict = torch.tensor([[59, 120, 137, 368, 0.124648176],
                                 [221, 89, 369, 367, 0.35818103],
                                 [54, 154, 148, 382, 0.13638769]]).cuda()
    # y= torch.randn(10000,5)
    start =time.time()
    for i in range(1000):
        x = non_maximum_suppression(predict_dict, 0.6, False)
    end = time.time()
    # print(x.cpu())
    print((end - start)/1000)
