from Chapter02.C02net import Net
import cv2
import torch


net = Net()
net.load_state_dict(torch.load('D:/data/chapter2/chapter02'))
"http://admin:admin@192.168.42.129:8081/video"
video = cv2.VideoCapture('5.gif')

def totensor(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)/255
    image = torch.tensor(image,dtype=torch.float32)
    return image.permute(2,0,1).unsqueeze(0)

while True:
    success, image = video.read()
    if success:
        image = cv2.resize(image, (300, 300))
        image_tensor = totensor(image)
        out = net(image_tensor)
        x1 = int(out[0][0].item() * 300)
        y1 = int(out[0][1].item() * 300)
        x2 = int(out[0][2].item() * 300)
        y2 = int(out[0][3].item() * 300)
        img = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("JK", img)
        if cv2.waitKey(100) == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()