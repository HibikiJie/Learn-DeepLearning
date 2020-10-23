from Chapter07.C07FastNeuralStyleTransfer import TransNet
import cv2
import torch
import numpy
net = TransNet()


net.load_state_dict(torch.load('D:/Learn-DeepLearning/Chapter07/fst.pth'))
net.eval().cuda()
def to_tensor(image):
    image = cv2.resize(image[:, :480], (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image.cuda()

def to_numpy(tensor):
    image = tensor.squeeze(0).permute(1,2,0)*255
    image = image.detach().cpu().numpy()
    image = image.astype(numpy.uint8)
    image = cv2.resize(image, (480, 480))
    return image

video_capture = cv2.VideoCapture("D:/data/chapter7/1.mp4")

while True:
    success, img = video_capture.read()
    image = net(to_tensor(img))
    image = to_numpy(image)
    cv2.imshow('Jk', image)
    if cv2.waitKey(1) == ord('q'):
        break

