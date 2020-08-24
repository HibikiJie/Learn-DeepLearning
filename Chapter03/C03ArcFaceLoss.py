from torch import nn
import torch
from torchvision.models import DenseNet
from Chapter03.test06 import MyArcCos


class ArcFaceLoss(nn.Module):

    def __init__(self, num_features, num_categories, angle):
        super(ArcFaceLoss, self).__init__()
        self.num_features = num_features
        self.num_categories = num_categories
        self.angle = angle
        self.acos = MyArcCos.apply
        self.w = nn.Parameter(torch.randn(self.num_features, self.num_categories))

    def forward(self, features, target):
        target = target.unsqueeze(dim=1)
        features = torch.nn.functional.normalize(features, dim=1)
        w = torch.nn.functional.normalize(self.w, dim=0)
        cos_theta = torch.matmul(features, w)
        theta = torch.acos(cos_theta)
        cos_theta_plus = torch.cos(theta+self.angle)
        top = torch.exp(cos_theta_plus).gather(dim=1, index=target)
        down_tempy = torch.exp(cos_theta)
        down = down_tempy.sum(dim=1, keepdim=True) - down_tempy.gather(dim=1, index=target) + top
        return -torch.log(top / down).sum() / len(target)


class ArcFace(nn.Module):

    def __init__(self, num_features, num_categories, angle=0.1, s=10):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.num_categories = num_categories
        self.angle = torch.tensor(angle)
        self.s = s
        self.w = nn.Parameter(torch.randn(self.num_features, self.num_categories))

    def forward(self, feature):
        feature = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(self.w, dim=0)
        cos_theta = torch.matmul(feature,w)/self.s
        theta = torch.acos(cos_theta)
        _top = torch.exp(self.s*torch.cos(theta))
        top = torch.exp(self.s*(torch.cos(theta + self.angle)))
        under = torch.sum(torch.exp(cos_theta*self.s),dim=1,keepdim=True)
        return top/(under - _top+top)

if __name__ == '__main__':
    arc_face_loss = ArcFace(2, 10)

    f = torch.randn(512, 2)
    print(f)
    label = torch.randint(10, (512,))
    loss = arc_face_loss(f)
    print(loss)
