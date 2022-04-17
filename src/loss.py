'''
Author: Ethan Chen
Date: 2022-04-08 07:37:13
LastEditTime: 2022-04-16 23:45:32
LastEditors: Ethan Chen
Description: 
FilePath: /CMPUT414/src/loss.py
'''
# classifier network
import torch.utils.data
import torch.nn.parallel
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# pointnet pytorch
# https://github.com/fxia22/pointnet.pytorch
# https://arxiv.org/abs/1612.00593


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        self.feature = []
        x = F.relu(self.bn1(self.conv1(x)))
        self.feature.append(x)  # 0
        x = F.relu(self.bn2(self.conv2(x)))
        self.feature.append(x)  # 1
        x = F.relu(self.bn3(self.conv3(x)))
        self.feature.append(x)  # 2
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        self.feature.append(x)  # 3
        x = F.relu(self.bn5(self.fc2(x)))
        self.feature.append(x)  # 4
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]
                                                  ).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x, self.feature


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[2]
        trans, self.feature = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        self.feature.append(x)  # 5
        trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        self.feature.append(x)  # 6
        x = self.bn3(self.conv3(x))
        self.feature.append(x)  # 7
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat, self.feature
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)
        x, trans, trans_feat, self.feature = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        self.feature.append(x)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        self.feature.append(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat, self.feature


class DGCNN(nn.Module):
    def __init__(self, k, emb_dims, dropout, output_channels=40):
        super(DGCNN, self).__init__()
        # self.args = args
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # print(x.shape)
        self.feature = []
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        self.feature.append(x)  # 0
        x = self.conv1(x)
        self.feature.append(x)  # 1
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        self.feature.append(x)  # 2
        x = self.conv2(x)
        self.feature.append(x)  # 3
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        self.feature.append(x)  # 4
        x = self.conv3(x)
        self.feature.append(x)  # 5
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        self.feature.append(x)  # 6
        x = self.conv4(x)
        self.feature.append(x)  # 7
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        self.feature.append(x)  # 8
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, self.feature


class PointNetLoss(nn.Module):
    def __init__(self, model, device, alpha=0.7):
        super(PointNetLoss, self).__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.model.to(self.device)
        self.model.eval()
        # self.feature_need = [5, 6, 7]
        self.feature_need = [0, 1, 2, 5, 6, 7]
        for parm in self.model.parameters():
            parm.requires_grad = False

    def forward(self, pointcloud_1, pointcloud_2):
        _, _, _, feature_map = self.model(pointcloud_1.to(self.device))
        _, _, _, feature_map_2 = self.model(pointcloud_2.to(self.device))
        loss = 0.0
        for i in self.feature_need:
            loss += self.alpha*F.l1_loss(feature_map[i], feature_map_2[i])
        return loss


class DGCNNLoss(nn.Module):
    def __init__(self, model, device, alpha=None):
        super(DGCNNLoss, self).__init__()
        self.model = model
        self.device = device
        self.alpha = alpha
        self.model.to(self.device)
        self.model.eval()
        self.feature_need = [1, 3, 5, 7, 8]
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1]
        for parm in self.model.parameters():
            parm.requires_grad = False

    def forward(self, pc1, pc2):
        _, feature_map = self.model(pc1.to(self.device))
        _, feature_map_2 = self.model(pc2.to(self.device))
        loss = 0
        for i in range(len(self.feature_need)):
            loss += self.weights[i]*F.l1_loss(feature_map[i], feature_map_2[i])
        return loss
