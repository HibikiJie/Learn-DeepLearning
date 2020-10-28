from Chapter07.C07FastNeuralStyleTransfer import TransNet
import cv2
import torch
import numpy
import imageio

def create_gif(image_list, gif_name, duration=1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    imageio.mimsave(gif_name, image_list, 'GIF', duration=duration)
    return
net = TransNet()


net.load_state_dict(torch.load('D:/Learn-DeepLearning/Chapter07/fst.pth'))
net.eval().cuda()
def to_tensor(image):
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image.cuda()

def to_numpy(tensor,h,w):
    image = tensor.squeeze(0).permute(1, 2, 0)*255
    image = image.detach().cpu().numpy()
    image = image.astype(numpy.uint8)
    image = cv2.resize(image, (w, h))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def save_mp4(path, images, fps):
    h, w, c = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for image in images:
        output_movie.write(image)

video_capture = cv2.VideoCapture("D:/data/chapter7/1.mp4")
images = []
while True:
    success, img = video_capture.read()
    if success:
        h, w, c = img.shape
        image = net(to_tensor(img))
        image = to_numpy(image,h,w)
        cv2.imshow('Jk', image)
        images.append(image)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
save_mp4('fww.mp4',images,50)


