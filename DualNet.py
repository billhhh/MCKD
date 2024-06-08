import torch
import torch.nn as nn
import torch.nn.functional as F
import Res50
from dsn import DomainSpecificNorm3d as DSN_Layer

import torch.distributions as tdist
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

visualization = False
tsne_points = []
tsne_labels = []
tsne_colors = []

ft_dist_switch = False


def plot_embedding_2D(data, label, colors, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(colors[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plot_embedding_3D(data, label, title):
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
    data = (data- x_min) / (x_max - x_min)
    #ax = plt.figure().add_subplot(111,projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9})
    return ax


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
              bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
        super(SeparableConv3d, self).__init__()
        self.depthwise = conv3x3x3(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=bias, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, in_channels)
        self.pointwise = conv3x3x3(in_channels, out_channels, kernel_size=1, bias=bias, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        x = self.pointwise(x)
        x = self.norm2(x)
        x = self.nonlin(x)
        return x


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, norm_cfg, activation_cfg, weight_std=False):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch2 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=False,
                      weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch3 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=4, dilation=4, bias=False,
                      weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch4 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=8, dilation=8, bias=False,
                      weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch5_conv = conv3x3x3(dim_in, dim_out, kernel_size=1, bias=False, weight_std=weight_std)
        self.branch5_norm = Norm_layer(norm_cfg, dim_out)
        self.branch5_nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv_cat = nn.Sequential(
            conv3x3x3(dim_out * 5, dim_out, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )

    def forward(self, x):
        [b, c, d, w, h] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, [2, 3, 4], True)
        global_f = global_feature
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_norm(global_feature)
        global_feature = self.branch5_nonlin(global_feature)
        global_feature = F.interpolate(global_feature, (d, w, h), None, 'trilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result, global_f


class SepResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(SepResBlock, self).__init__()
        self.sepconv1 = SeparableConv3d(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                        bias=False, weight_std=weight_std)
        self.sepconv2 = SeparableConv3d(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                        bias=False, weight_std=weight_std)
        # self.sepconv3 = SeparableConv3d(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.sepconv1(x)
        out = self.sepconv2(out)
        # out = self.sepconv3(out)
        out = out + residual

        return out


class U_Res3D_enc(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='LeakyReLU', num_classes=None, weight_std=False):
        super(U_Res3D_enc, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.asppreduce = nn.Sequential(
            conv3x3x3(1280, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.aspp = ASPP(256, 256, norm_cfg, activation_cfg, weight_std=weight_std)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = Res50.ResNet(depth=50, shortcut_type='B', norm_cfg=norm_cfg, activation_cfg=activation_cfg,
                                     weight_std=weight_std)

    def forward(self, x):
        _ = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_reducechannel = self.asppreduce(layers[-1])
        feature_aspp, global_f = self.aspp(feature_reducechannel)  # 256

        return feature_aspp, global_f


class U_Res3D_dec(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='LeakyReLU', num_classes=None, weight_std=False):
        super(U_Res3D_dec, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.upsamplex2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        self.shortcut_conv3 = nn.Sequential(
            conv3x3x3(256 * 16, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv2 = nn.Sequential(
            conv3x3x3(128 * 16, 128, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 128),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv1 = nn.Sequential(
            conv3x3x3(64 * 16, 64, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 64),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv0 = nn.Sequential(
            conv3x3x3(64 * 4, 32, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 32),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.transposeconv_stage3 = nn.ConvTranspose3d(256 * 4, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

        # self.stage2_de = SepResBlock(256, 256, norm_cfg, activation_cfg, weight_std=weight_std)
        # self.stage1_de = SepResBlock(128, 128, norm_cfg, activation_cfg, weight_std=weight_std)
        # self.stage0_de = SepResBlock(32, 32, norm_cfg, activation_cfg, weight_std=weight_std)

        self.stage3_de = Res50.BasicBlock(256, 256, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage2_de = Res50.BasicBlock(128, 128, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = Res50.BasicBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = Res50.BasicBlock(32, 32, norm_cfg, activation_cfg, weight_std=weight_std)

        self.cls_conv = nn.Sequential(
            nn.Conv3d(32, self.MODEL_NUM_CLASSES, kernel_size=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, layers):
        x = self.transposeconv_stage3(x)
        skip3 = self.shortcut_conv3(layers[-2])
        x = x + skip3
        x = self.stage3_de(x)

        x = self.transposeconv_stage2(x)
        skip2 = self.shortcut_conv2(layers[-3])
        x = x + skip2
        x = self.stage2_de(x)

        x = self.transposeconv_stage1(x)
        skip1 = self.shortcut_conv1(layers[-4])
        x = x + skip1
        x = self.stage1_de(x)

        x = self.transposeconv_stage0(x)
        skip0 = self.shortcut_conv0(layers[-5])
        x = x + skip0
        x = self.stage0_de(x)
        self.x_stage0_de = x

        logits = self.cls_conv(x)
        logits = self.upsamplex2(logits)
        return [logits]


class KD_Weights(nn.Module):
    def __init__(self, shape):
        super(KD_Weights, self).__init__()
        self.kd_weights = nn.Parameter(torch.rand(shape), requires_grad=True)  # kd weights as teachers
        # self.kd_weights = nn.Parameter(torch.ones(shape), requires_grad=True)

    def get_kdWeights(self):
        # return self.kd_weights
        # return torch.sigmoid(self.kd_weights)
        return F.softmax(self.kd_weights)

    def forward(self, i, j):
        # kd_w = self.kd_weights[i] / self.kd_weights[j]
        # kd_w = torch.sigmoid(self.kd_weights[i]) / torch.sigmoid(self.kd_weights[j])
        kd_w = F.softmax(self.kd_weights)[i]/F.softmax(self.kd_weights)[j]
        return kd_w


# Meta-Learning
class DualNet(nn.Module):
    def __init__(self, args, norm_cfg='BN', activation_cfg='LeakyReLU', num_classes=None,
                 weight_std=False, self_att=False, cross_att=False):
        super().__init__()
        self.do_ds = False
        self.shared_enc = U_Res3D_enc(norm_cfg, activation_cfg, num_classes, weight_std)
        self.shared_dec = U_Res3D_dec(norm_cfg, activation_cfg, num_classes, weight_std)

        # MHA
        self.self_att = self_att
        self.cross_att = cross_att
        embed_dim = 125
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=5).cuda()

        self.kd_loss = nn.L1Loss(reduce=False)
        # self.kd_loss = nn.MSELoss(reduce=False)

        self.kd_weights = KD_Weights(shape=4)

    """
    mode: which modality/modalities are available
    """
    def forward(self, images, val=False, mode='0,1,2,3'):
        org_mode = mode
        N, C, D, W, H = images.shape
        if val:
            mode = '0,1,2,3'

        # random choose modes instead of specifying
        if org_mode == 'random':
            mode = '0,1,2,3'
            mode_num = random.randint(1, 4)
        mode_split = mode.split(',')
        mode_split = list(map(int, mode_split))
        if org_mode == 'random':
            mode_split = random.sample(mode_split, mode_num)

        # delete modality
        _images = images.clone()
        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                _images[:, m, :, :, :] = 0

        # a=torch.Tensor([[[1,2,3,4],[5,6,7,8]]])
        # print(a.view(8, 1))
        x = _images.view(N * C, 1, D, W, H)
        shared_ft, shared_gft = self.shared_enc(x)
        shared_layers = self.shared_enc.backbone.get_layers()
        _shared_layers = [i_layer.view(N, C * i_layer.shape[1], i_layer.shape[2], i_layer.shape[3], i_layer.shape[4])
                          for i_layer in shared_layers]

        # KD process
        fused_ft = shared_ft.clone()
        flair_ft = shared_ft[0:N * C + 1:C]
        t1_ft = shared_ft[1:N * C + 1:C]
        t1ce_ft = shared_ft[2:N * C + 1:C]
        t2_ft = shared_ft[3:N * C + 1:C]

        # fts fill in process
        fill_fts = torch.zeros_like(shared_ft[0:N * C + 1:C], device='cuda')
        for m in mode_split:
            fill_fts += shared_ft[m:N * C + 1:C]
        fill_fts /= len(mode_split)

        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                fused_ft[m:N * C + 1:C] = fill_fts

        totKDLoss = torch.tensor(0.0, device='cuda')
        if val:
            mode_w = self.kd_weights.get_kdWeights()
            fused_ft[0:N * C + 1:C] = flair_ft * mode_w[0]
            fused_ft[1:N * C + 1:C] = t1_ft * mode_w[1]
            fused_ft[2:N * C + 1:C] = t1ce_ft * mode_w[2]
            fused_ft[3:N * C + 1:C] = t2_ft * mode_w[3]

        if self.self_att:
            # self attention
            cat_attend = fused_ft.view(N, C * fused_ft.shape[1], fused_ft.shape[2], fused_ft.shape[3],
                                       fused_ft.shape[4])
            original_size = cat_attend.size()
            flat_input = cat_attend.view(original_size[0], original_size[1],
                                         original_size[2] * original_size[3] * original_size[4])
            perm_input = flat_input.permute(1, 0, 2)
            att_input, att_weights = self.multihead_attn(perm_input, perm_input, perm_input)

            flat_output = att_input.permute(1, 0, 2)
            out_ft = flat_output.view(original_size)
        else:
            out_ft = fused_ft.view(N, C * fused_ft.shape[1], fused_ft.shape[2], fused_ft.shape[3], fused_ft.shape[4])

        if not val:
            for i in [0, 1, 2, 3]:  # teachers
                if i not in mode_split: continue
                for j in [0, 1, 2, 3]:  # students
                    if i == j or j not in mode_split: continue
                    kd_w = self.kd_weights(i, j)
                    totKDLoss = totKDLoss + kd_w.detach() * self.kd_loss(shared_ft[i:N * C + 1:C].detach(), shared_ft[j:N * C + 1:C]).mean(dim=(1, 2, 3, 4))
        print("kd_weights", self.kd_weights.kd_weights)

        # segmentation
        logits = self.shared_dec(out_ft, _shared_layers)

        # for analysis
        if val and ft_dist_switch:
            self.print_ft_dist(images)
        return logits[0], mode_split, totKDLoss.mean()

    def print_ft_dist(self, images, mode='0,1,3'):
        N, C, D, W, H = images.shape
        mode_split = mode.split(',')
        mode_split = list(map(int, mode_split))

        # delete modality
        _images = images.clone()
        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                _images[:, m, :, :, :] = 0

        x = _images.view(N * C, 1, D, W, H)
        shared_ft, _ = self.shared_enc(x)
        ft_incomplete = shared_ft.detach()

        # fts fill in process
        fill_fts = torch.zeros_like(ft_incomplete[0:N * C + 1:C], device='cuda')
        for m in mode_split:
            fill_fts += ft_incomplete[m:N * C + 1:C]
        fill_fts /= len(mode_split)

        x_all = images.view(N * C, 1, D, W, H)
        fts, _ = self.shared_enc(x_all)
        ft_all = fts.detach()
        ft_l1_dist = 0
        ft_l2_dist = 0
        ft_cos_sim = 0
        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                ft_l1_dist += abs(fill_fts - ft_all[m:N * C + 1:C]).sum()
                ft_l2_dist += torch.norm(fill_fts - ft_all[m:N * C + 1:C], 2)
                ft_cos_sim += F.cosine_similarity(fill_fts.view(1, -1), ft_all[m:N * C + 1:C].view(1, -1), dim=1)[0]
        print('Feature L1 distance:', ft_l1_dist)
        print('Feature L2 distance:', ft_l2_dist)
        print('Feature Cos Similarity:', ft_cos_sim)
