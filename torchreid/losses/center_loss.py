import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    """Center Loss
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016
    Args:
        num_classes: 训练集行人ID总数
        feat_dim: 特征维度（ResNet50为2048）
        lambda_c: 内部权重缩放，外部用weight_c控制总权重
    """
    def __init__(self, num_classes, feat_dim, lambda_c=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        # 可学习类别中心向量
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, targets):
        batch_size = features.size(0)
        # 取出当前batch对应类别的中心
        centers_batch = self.centers[targets]
        # 计算特征与对应类别中心的L2距离平方
        dist_sq = torch.sum(torch.pow(features - centers_batch, 2), dim=1)
        loss = torch.sum(dist_sq) / (2.0 * batch_size) * self.lambda_c
        return loss