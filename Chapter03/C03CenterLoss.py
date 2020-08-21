from torch import nn
import torch


class CenterLoss(nn.Module):

    def __init__(self, feature_number, class_number):
        """
        初始化中心损失(center loss)
        :param feature_number:特征的数量
        :param class_number:分类的数量
        """
        super(CenterLoss, self).__init__()
        self.class_number = class_number
        self.feature_number = feature_number

        '''将中心点定义为可以学习的参数'''
        self.center_point = nn.Parameter(torch.randn(class_number,feature_number), requires_grad=True)
        '''
            [1,0],
            [0.809016994374947424,0.587785252292],
            [0.309016994375, 0.951056516295],
            [-0.309016994375, 0.951056516295],
            [-0.809016994375, 0.587785252292],
            [-1, 0],
            [-0.809016994375, -0.587785252292],
            [-0.309016994375, -0.951056516295],
            [0.309016994375, -0.951056516295],
            [0.809016994374947424, -0.587785252292],
        '''
    def forward(self, feature, target):
        """
        计算中心损失
        :param feature: 特征向量。
        :param target: 分类标签，未进行one_hot处理的。
        :return: 损失
        """
        '''通过标签，将每个点与各自对于的类别的中心点对应'''
        midpoint_tensor = self.center_point.index_select(dim=0, index=target.long())

        '''统计每个类别的数量'''
        target_hist = torch.histc(target.float(), self.class_number, min=0, max=self.class_number - 1)

        '''通过标签，将每个点的类别的数量与特征向量对应'''
        hist_tensor = target_hist.index_select(dim=0, index=target.long())

        '''计算损失，先计算多维空间中两点的欧式距离，再除去自身类别数，即平均。最后加和，求得总损失。'''
        return torch.sum(torch.sqrt(torch.sum(torch.pow(feature-midpoint_tensor,2),dim=1))/hist_tensor)
