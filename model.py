from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        self.conv1 = nn.Sequential(
            OrderedDict([('conv1_a',
                          nn.Conv2d(
                              1,
                              16,
                              kernel_size=(3, 7),
                              stride=1,
                              padding=(1, 3),
                              bias=False)), ('tan1_a', nn.Tanh())]))
        self.conv2 = nn.Sequential(
            OrderedDict([('pool1_a', nn.AvgPool2d(kernel_size=2, stride=2)),
                         ('conv2_a',
                          nn.Conv2d(
                              16,
                              32,
                              kernel_size=(3, 5),
                              stride=1,
                              padding=(1, 2),
                              bias=False)), ('tan2_a', nn.Tanh())]))
        self.conv3 = nn.Sequential(
            OrderedDict([('pool2_a', nn.AvgPool2d(kernel_size=2, stride=2)),
                         ('conv3_a',
                          nn.Conv2d(
                              32,
                              64,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)), ('tan3_a', nn.Tanh())]))
        self.fuse_a = nn.Conv2d(
            112, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x2 = F.interpolate(
            x2, size=(64, 512), mode='bilinear', align_corners=False)
        x3 = F.interpolate(
            x3, size=(64, 512), mode='bilinear', align_corners=False)
        x4 = torch.cat((x1, x2, x3), dim=1)
        out = self.fuse_a(x4)
        return out


class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet, self).__init__()
        self.m_conv1 = nn.Sequential(
            OrderedDict([
                ('m_conv1_a',
                 nn.Conv2d(
                     1, 16, kernel_size=5, stride=1, padding=2, bias=True)),
                ('m_relu_a', nn.ReLU(inplace=True)),
            ]))
        self.m_conv2 = nn.Sequential(
            OrderedDict([('m_pool1_a', nn.MaxPool2d(kernel_size=2, stride=2)),
                         ('m_conv2_a',
                          nn.Conv2d(
                              16,
                              32,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)), ('m_relu2_a',
                                            nn.ReLU(inplace=True))]))
        self.m_score2_a = nn.Sequential(
            OrderedDict([('m_score2_a',
                          nn.Conv2d(32, 2, kernel_size=1, stride=1,
                                    bias=True))]))
        self.m_conv3 = nn.Sequential(
            OrderedDict([('m_pool2_a', nn.MaxPool2d(kernel_size=2, stride=2)),
                         ('m_conv3_a',
                          nn.Conv2d(
                              32,
                              64,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)), ('m_relu3_a',
                                            nn.ReLU(inplace=True))]))
        self.m_score3_a = nn.Sequential(
            OrderedDict([('m_score3_a',
                          nn.Conv2d(64, 2, kernel_size=1, stride=1,
                                    bias=True))]))
        self.m_conv4 = nn.Sequential(
            OrderedDict([('m_pool3_a', nn.MaxPool2d(kernel_size=4, stride=4)),
                         ('m_conv4_a',
                          nn.Conv2d(
                              64,
                              128,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)), ('m_relu4_a',
                                            nn.ReLU(inplace=True))]))
        self.m_score4_a = nn.Sequential(
            OrderedDict([('m_score4_a',
                          nn.Conv2d(
                              128, 2, kernel_size=1, stride=1, bias=True))]))

    def forward(self, x):
        x1 = self.m_conv2(self.m_conv1(x))
        x2 = self.m_conv3(x1)
        x3 = self.m_score4_a(self.m_conv4(x2))
        x34 = self.m_score3_a(x2) + F.interpolate(
            x3, size=(16, 128), mode='bilinear', align_corners=False)
        x234 = self.m_score2_a(x1) + F.interpolate(
            x34, size=(32, 256), mode='bilinear', align_corners=False)
        out = F.interpolate(
            x234, size=(64, 512), mode='bilinear', align_corners=False)
        return out


class mfm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(
                in_channels,
                2 * out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class Maxout_4(nn.Module):
    def __init__(self, num_classes):
        super(Maxout_4, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 9, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(48, 96, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(96, 128, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(128, 192, 4, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(5 * 5 * 192, 256, type=0)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feature = self.fc1(x)
        x = self.dropout(feature)
        out = self.fc2(x)
        return out, feature


class Maxout_feature(nn.Module):
    def __init__(self):
        super(Maxout_feature, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 9, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(48, 96, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(96, 128, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(128, 192, 4, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(5 * 5 * 192, 256, type=0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feature = F.sigmoid(self.fc1(x))
        return feature


class FaceModel(nn.Module):
    def __init__(self, embedding_size, pretrained=False):
        super(FaceModel, self).__init__()

        self.model = resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(26112, self.embedding_size)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        features = features * alpha

        return features


if __name__ == '__main__':
    a = torch.rand((1, 1, 64, 512))
    f = FeatNet()
    print(f(a).shape)
    m = MaskNet()
    print(m(a).shape)
