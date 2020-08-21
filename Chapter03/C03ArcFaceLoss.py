from torch import nn
import torch
from torchvision.models import DenseNet

class ArcFaceLoss(nn.Module):

    def __init__(self, num_features, num_categories, angle):
        super(ArcFaceLoss, self).__init__()
        self.num_features = num_features
        self.num_categories = num_categories
        self.angle = angle

        self.w = nn.Parameter(torch.randn(self.num_features,self.num_categories))

    def forward(self,features,target):
        target = target.unsqueeze(dim=1)
        features_modulus = torch.sum(features ** 2, dim=1, keepdim=True) ** 0.5
        w_modulus = torch.sum(self.w**2,dim=0,keepdim=True)**0.5
        modulus = features_modulus*w_modulus
        cos_theta = torch.matmul(features, self.w)/modulus/1.01
        print(cos_theta.max())
        theta = torch.acos(cos_theta)+self.angle
        cos_theta_plus = torch.cos(theta)
        top = torch.exp(modulus*cos_theta_plus).gather(dim=1,index=target)
        down_ = torch.exp(torch.matmul(features, self.w))
        down = down_.sum(dim=1,keepdim=True)-down_.gather(dim=1,index=target)+top
        return -torch.log(top/down).sum()/len(target)

if __name__ == '__main__':
    arc_face_loss = ArcFaceLoss(2, 10,0.2)

    f = torch.randn(512,2)
    print(f)
    label = torch.randint(10,(512,))
    loss = arc_face_loss(f,label)
    print(loss)



