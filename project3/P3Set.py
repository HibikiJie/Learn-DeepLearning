import numpy
class Set:

    def __init__(self):
        self.boxes_base = {
            52: ((29, 40), (15, 17), (65, 55)),
            26: ((49, 105), (88, 176), (123, 99)),
            13: ((311, 277), (158, 238), (262, 144))
        }
        self.boxes_base2 = {
            52: numpy.array([[29, 40], [15, 17], [65, 55]], dtype=numpy.float32),
            26: numpy.array([[49, 105], [88, 176], [123, 99]], dtype=numpy.float32),
            13: numpy.array([[311, 277], [158, 238], [262, 144]], dtype=numpy.float32)
        }
        self.threshold = 0.99
        self.category = ['person', 'head', 'hand', 'foot', 'aeroplane', 'tvmonitor', 'train', 'boat', 'dog', 'chair', 'bird',
            'bicycle', 'bottle', 'sheep', 'diningtable', 'horse', 'motorbike', 'sofa', 'cow', 'car', 'cat', 'bus',
            'pottedplant']
        self.num_category = len(self.category)
        self.iou_threshold = 0.1

