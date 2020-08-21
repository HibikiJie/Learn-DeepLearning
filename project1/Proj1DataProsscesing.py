import os
import random
from PIL import Image
import tqdm


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
    x_1 = max([x1, x3])
    y_1 = max([y1, y3])
    x_2 = min([x2, x4])
    y_2 = min([y2, y4])

    '''计算相交区域的面积'''
    intersection_area = (max([0, x_2 - x_1])) * (max([0, y_2 - y_1]))

    '''计算两候选框各自的面积'''
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    '''返回交并比'''
    return intersection_area / (area1 + area2 - intersection_area)


def compute_iou_min(box1, box2):
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
    x_1 = max([x1, x3])
    y_1 = max([y1, y3])
    x_2 = min([x2, x4])
    y_2 = min([y2, y4])

    '''计算相交区域的面积'''
    intersection_area = (max([0, x_2 - x_1])) * (max([0, y_2 - y_1]))

    '''计算两候选框各自的面积'''
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    '''返回交并比'''
    return intersection_area / min(area1, area2)


'''原始文件地址'''
origin_image_path = 'D:/data/object1/img_celeba'
origin_txt_path = 'D:/data/object1/list_bbox_celeba.txt'

'''文件的存储地址'''
save_path = 'c:/data/'

'''读取标注信息'''
with open(origin_txt_path, 'r') as file:
    txt = file.read().split()[6:]
datas = []
for i in range(len(txt) // 5):
    datas.append(txt[5 * i:5 * i + 5])

'''生成不同尺寸大小的图片'''
for face_size in [24]:
    print('生成 %i x %i 的图片' % (face_size, face_size))

    '''图片的存储地址'''
    positive_image_path = '{}/{}/{}'.format(save_path, face_size, 'positive')
    part_image_path = '{}/{}/{}'.format(save_path, face_size, 'part')
    negative_image_path = '{}/{}/{}'.format(save_path, face_size, 'negative')

    '''创建文件夹'''
    for dir in [positive_image_path, part_image_path, negative_image_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    '''各类型图片数据的标注信息存储文件路径'''
    positive_txt_path = '{}/{}/{}'.format(save_path, face_size, 'positive.txt')
    part_txt_path = '{}/{}/{}'.format(save_path, face_size, 'part.txt')
    negative_txt_path = '{}/{}/{}'.format(save_path, face_size, 'negative.txt')

    '''生成样本的计数器'''
    positive_count = 0
    negative_count = 0
    part_count = 0

    '''打开文件'''
    positive_file = open(positive_txt_path, 'w')
    part_file = open(part_txt_path, 'w')
    negative_file = open(negative_txt_path, 'w')

    '''遍历所有图片'''
    for image_data in tqdm.tqdm(datas):
        image_name = image_data[0]  # 取出文件名
        image_path = f'{origin_image_path}/{image_name}'  # 生成寻址路径

        '''取出标记框'''
        x1 = int(image_data[1])
        y1 = int(image_data[2])
        w = int(image_data[3])
        h = int(image_data[4])
        x2 = x1 + w
        y2 = y1 + h
        box = [[x1, y1], [x2, y2]]  # 组合成box框

        '''过滤掉异常数据'''
        if max(w, h) < 48 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
            continue

        '''生成正样本'''
        count = 0  # 生成数量计数器
        try_count = 0
        while count < 2:
            '''求得人脸标注框的中心点'''
            center_x = x1 + w // 2
            center_y = y1 + h // 2
            # print(center_x, center_y)

            '''在中心点基础上，随机偏移'''
            center_x = center_x + random.randint(int(-w * 0.2), int(w * 0.2))
            center_y = center_y + random.randint(int(-h * 0.2), int(h * 0.2))

            '''随机扣取框的大小'''
            side_length = random.randint(int(min(w, h) * 0.8), int(max(w, h) * 1.1))

            '''计算抠取框的上下两点坐标'''
            x1_skim = center_x - side_length // 2
            y1_skim = center_y - side_length // 2
            x2_skim = center_x + side_length // 2
            y2_skim = center_y + side_length // 2

            '''组合，计算交并比'''
            box_skim = [[x1_skim, y1_skim], [x2_skim, y2_skim]]
            iou = compute_iou(box, box_skim)

            if iou > 0.7:
                new_image_name = str(positive_count).rjust(8, '0') + '.jpg'

                '''计算相对坐标的位置'''
                x1_relative = (x1 - x1_skim) / side_length
                y1_relative = (y1 - y1_skim) / side_length
                x2_relative = (x2 - x2_skim) / side_length
                y2_relative = (y2 - y2_skim) / side_length

                txt_information = f"{new_image_name}  {x1_relative}  {y1_relative}  {x2_relative}  {y2_relative}\n"
                positive_file.write(txt_information)
                image = Image.open(image_path)
                image_crop = image.crop((x1_skim, y1_skim, x2_skim, y2_skim))
                image_resize = image_crop.resize((face_size, face_size), Image.ANTIALIAS)
                image_resize.save(f'{positive_image_path}/{new_image_name}')
                # print(txt_information)
                positive_count += 1
                count += 1
            try_count += 1
            if try_count >= 50:
                break

        '''生成部分样本'''
        count = 0  # 生成数量计数器
        try_count = 0
        while count < 2:
            '''求得人脸标注框的中心点'''
            center_x = x1 + w // 2
            center_y = y1 + h // 2
            # print(center_x, center_y)

            '''在中心点基础上，随机偏移'''
            center_x = center_x + random.randint(int(-w * 0.2), int(w * 0.2))
            center_y = center_y + random.randint(int(-h * 0.2), int(h * 0.2))

            '''随机扣取框的大小'''
            side_length = random.randint(int(min(w, h) * 0.8), int(max(w, h) * 1.1))

            '''计算抠取框的上下两点坐标'''
            x1_skim = center_x - side_length // 2
            y1_skim = center_y - side_length // 2
            x2_skim = center_x + side_length // 2
            y2_skim = center_y + side_length // 2

            '''组合，计算交并比'''
            box_skim = [[x1_skim, y1_skim], [x2_skim, y2_skim]]
            iou = compute_iou(box, box_skim)

            if 0.4 < iou < 0.6:
                new_image_name = str(part_count).rjust(8, '0') + '.jpg'

                '''计算相对坐标的位置'''
                x1_relative = (x1 - x1_skim) / side_length
                y1_relative = (y1 - y1_skim) / side_length
                x2_relative = (x2 - x2_skim) / side_length
                y2_relative = (y2 - y2_skim) / side_length

                txt_information = f"{new_image_name}  {x1_relative}  {y1_relative}  {x2_relative}  {y2_relative}\n"
                part_file.write(txt_information)
                image = Image.open(image_path)
                image_crop = image.crop((x1_skim, y1_skim, x2_skim, y2_skim))
                image_resize = image_crop.resize((face_size, face_size), Image.ANTIALIAS)
                image_resize.save(f'{part_image_path}/{new_image_name}')
                # print(txt_information)
                part_count += 1
                count += 1
            try_count += 1
            if try_count >= 50:
                break
        '''生成负样本'''
        count = 0
        try_count = 0
        while count < 10:
            '''求得人脸标注框的中心点'''
            image = Image.open(image_path)
            width, height = image.size
            side_length = random.randint(face_size, max(int(min(w, h) * 0.5), face_size + 1))
            x1_skim = random.randint(0, width - side_length)
            y1_skim = random.randint(0, height - side_length)
            x2_skim = x1_skim + side_length
            y2_skim = y1_skim + side_length
            box_skim = [[x1_skim, y1_skim], [x2_skim, y2_skim]]
            iou = compute_iou_min(box, box_skim)
            if iou < 0.2:
                new_image_name = str(negative_count).rjust(8, '0') + '.jpg'

                txt_information = f"{new_image_name}  0  0  0  0\n"
                negative_file.write(txt_information)
                image = Image.open(image_path)
                image_crop = image.crop((x1_skim, y1_skim, x2_skim, y2_skim))
                image_resize = image_crop.resize((face_size, face_size), Image.ANTIALIAS)
                image_resize.save(f'{negative_image_path}/{new_image_name}')
                # print(txt_information)
                negative_count += 1
                count += 1
            try_count += 1
            if try_count >= 50:
                break
