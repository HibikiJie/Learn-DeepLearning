import numpy


def compute_iou(box1, box2):
    """
    计算两个候选框之间的交并比
    :param box1: 候选框的两个坐标值
    :param box2: 候选框的两个坐标值
    :return:
        iou:iou of box1 and box2
    """
    '''分别获取坐标'''
    x1, y1 = box1[0]
    x2, y2 = box1[1]
    x3, y3 = box2[0]
    x4, y4 = box2[1]

    '''计算两候选框之间的矩形相交区域的两角点的坐标'''
    x_1 = numpy.max([x1, x3])
    y_1 = numpy.max([y1, y3])
    x_2 = numpy.min([x2, x4])
    y_2 = numpy.min([y2, y4])

    '''计算相交区域的面积'''
    intersection_area = (numpy.max([0, x_2 - x_1])) * (numpy.max([0, y_2 - y_1]))

    '''计算两候选框各自的面积'''
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    '''返回交并比'''
    return intersection_area / (area1 + area2 - intersection_area)


if __name__ == '__main__':
    box1 = numpy.array([[20, 30], [50, 60]])
    box2 = numpy.array([[40, 50], [60, 70]])
    print(compute_iou(box1, box2))
    print(compute_iou(box2, box1))
    box1 = numpy.array([[20, 30], [50, 60]])
    box2 = numpy.array([[90, 80], [120, 100]])
    print(compute_iou(box1, box2))
    print(compute_iou(box2, box1))
    box1 = numpy.array([[20, 30], [50, 60]])
    box2 = numpy.array([[20, 30], [50, 60]])
    print(compute_iou(box1, box2))
    print(compute_iou(box2, box1))



