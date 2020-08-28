from torchvision.models import densenet121,densenet161
from torch import nn
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.net = densenet121(True)
        self.net.classifier = nn.Linear(1024, 1000,False)

    def forward(self, input_):
        return self.net(input_)


class ArcFace(nn.Module):

    def __init__(self, num_features, num_categories, angle=0.1, s=10.):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.num_categories = num_categories
        self.angle = torch.tensor(angle)
        self.s = s
        self.w = nn.Parameter(torch.randn(self.num_features, self.num_categories))

    def forward(self, feature):
        feature = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(self.w, dim=0)
        cos_theta = torch.matmul(feature, w) / self.s
        theta = torch.acos(cos_theta)
        _top = torch.exp(self.s * torch.cos(theta))
        top = torch.exp(self.s * (torch.cos(theta + self.angle)))
        under = torch.sum(torch.exp(cos_theta * self.s), dim=1, keepdim=True)
        return top / (under - _top + top), cos_theta.detach().cpu()
