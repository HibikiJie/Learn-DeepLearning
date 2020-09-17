from Chapter06.C6识别验证码 import GRUNet
import torch
import cv2
import os
codes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
         'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def test(path):
    net = GRUNet().eval()
    net.load_state_dict(torch.load('D:/data/chapter6/net.pth'))
    for image_name in os.listdir(path):
        image_path = f'{path}/{image_name}'
        img = cv2.imread(image_path)
        image = torch.from_numpy(img).float()
        image = image / 255
        output = net(image.unsqueeze(0))
        output = output.reshape(-1, 4, 36)
        output = output.argmax(2)[0]
        code = ''
        for i in output:
            code += codes[i]
        cv2.imshow(code, img)
        print(image_name[:4],code)
        if cv2.waitKey() == ord('n'):
            continue


if __name__ == '__main__':
    test('D:/data/chapter6/test')