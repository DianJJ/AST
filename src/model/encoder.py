from copy import deepcopy
import torch
from torch import nn
from torchvision import models as tv_models
from .resnet_dilated import ResnetDilated
from .tools import get_output_size, Identity, kaiming_weights_init
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

def get_resnet_model(name):
    if name is None:
        name = 'resnet18'
    return {
        'resnet18': tv_models.resnet18,
        'resnet34': tv_models.resnet34,
        'resnet50': tv_models.resnet50,
        'resnet101': tv_models.resnet101,
        'resnet152': tv_models.resnet152,
        'resnext50_32x4d': tv_models.resnext50_32x4d,
        'resnext101_32x8d': tv_models.resnext101_32x8d,
        'wide_resnet50_2': tv_models.wide_resnet50_2,
        'wide_resnet101_2': tv_models.wide_resnet101_2,
    }[name]

class Encoder(nn.Module):
    def __init__(self, out_channels=128, dims=16, nres_block=2, normalizer_fn=None, demosaic=False, use_center=False,
                 use_noise_map=False):
        super(Encoder, self).__init__()

        self.out_channels = out_channels
        self.dims = dims
        self.nres_block = nres_block
        self.normalizer_fn = normalizer_fn
        self.demosaic = demosaic
        self.use_center = use_center
        self.use_noise_map = use_noise_map
        self.out_ch=128

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dims, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims, out_channels=dims * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=dims * 2, out_channels=dims * 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=dims * 4, out_channels=dims * 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=dims * 8, out_channels=dims * 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=dims * 16, out_channels=dims * 12, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=dims * 12, out_channels=dims * 10, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=dims * 10, out_channels=dims*9, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=dims*9, out_channels=out_channels, kernel_size=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, inputs):

        conv1 = F.leaky_relu(self.conv1(inputs))
        conv2 = F.leaky_relu(self.conv2(conv1))
        pool1 = F.max_pool2d(conv2, kernel_size=2)
        conv3 = F.leaky_relu(self.conv3(pool1))
        pool2 = F.max_pool2d(conv3, kernel_size=2)
        conv4 = F.leaky_relu(self.conv4(pool2))
        pool3 = F.max_pool2d(conv4, kernel_size=2)
        conv5 = F.leaky_relu(self.conv5(pool3))
        conv6 = F.leaky_relu(self.conv6(conv5))

        up7 = F.interpolate(conv6, scale_factor=2, mode="nearest")
        up7 = self.conv7(up7)
        up7 = torch.cat([up7, conv4], dim=1)
        conv7 = F.leaky_relu(self.conv7(up7))

        up8 = F.interpolate(conv7, scale_factor=2, mode="nearest")
        up8 = self.conv8(up8)
        up8 = torch.cat([up8, conv3], dim=1)
        conv8 = F.leaky_relu(self.conv8(up8))

        up9 = F.interpolate(conv8, scale_factor=2, mode="nearest")
        up9 = self.conv9(up9)
        up9 = torch.cat([up9, conv2], dim=1)
        conv9 = F.leaky_relu(self.conv9(up9))

        conv10 = self.conv10(conv9)
        out = conv10

        return out

class Encoder_32(nn.Module):
    def __init__(self, out_channels=128, dims=16, nres_block=2, normalizer_fn=None, demosaic=False, use_center=False,
                 use_noise_map=False):
        super(Encoder_32, self).__init__()

        self.out_channels = out_channels
        self.dims = dims
        self.nres_block = nres_block
        self.normalizer_fn = normalizer_fn
        self.demosaic = demosaic
        self.use_center = use_center
        self.use_noise_map = use_noise_map
        self.out_ch=128

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dims, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims, out_channels=dims * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=dims * 2, out_channels=dims * 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=dims * 4, out_channels=dims * 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=dims * 8, out_channels=dims * 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=dims * 16, out_channels=dims * 12, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=dims * 12, out_channels=dims * 10, kernel_size=3, padding=1)
        #self.conv9 = nn.Conv2d(in_channels=dims * 10, out_channels=dims*9, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=dims*10, out_channels=out_channels, kernel_size=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, inputs):

        conv1 = F.leaky_relu(self.conv1(inputs))
        conv2 = F.leaky_relu(self.conv2(conv1))
        pool1 = F.max_pool2d(conv2, kernel_size=2)
        conv3 = F.leaky_relu(self.conv3(pool1))
        pool2 = F.max_pool2d(conv3, kernel_size=2)
        conv4 = F.leaky_relu(self.conv4(pool2))
        pool3 = F.max_pool2d(conv4, kernel_size=2)
        conv5 = F.leaky_relu(self.conv5(pool3))
        conv6 = F.leaky_relu(self.conv6(conv5))

        up7 = F.interpolate(conv6, scale_factor=2, mode="nearest")
        up7 = self.conv7(up7)
        up7 = torch.cat([up7, conv4], dim=1)
        conv7 = F.leaky_relu(self.conv7(up7))

        up8 = F.interpolate(conv7, scale_factor=2, mode="nearest")
        up8 = self.conv8(up8)
        up8 = torch.cat([up8, conv3], dim=1)
        conv8 = F.leaky_relu(self.conv8(up8))

        conv9 = self.conv9(conv8)
        out = conv9
        return out

class Encoder_16(nn.Module):
    def __init__(self, out_channels=128, dims=16, nres_block=2, normalizer_fn=None, demosaic=False, use_center=False,
                 use_noise_map=False):
        super(Encoder_16, self).__init__()

        self.out_channels = out_channels
        self.dims = dims
        self.nres_block = nres_block
        self.normalizer_fn = normalizer_fn
        self.demosaic = demosaic
        self.use_center = use_center
        self.use_noise_map = use_noise_map
        self.out_ch=128

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dims, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims, out_channels=dims * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=dims * 2, out_channels=dims * 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=dims * 4, out_channels=dims * 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=dims * 8, out_channels=dims * 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=dims * 16, out_channels=dims * 12, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=dims*12, out_channels=out_channels, kernel_size=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, inputs):

        conv1 = F.leaky_relu(self.conv1(inputs))
        conv2 = F.leaky_relu(self.conv2(conv1))
        pool1 = F.max_pool2d(conv2, kernel_size=2)
        conv3 = F.leaky_relu(self.conv3(pool1))
        pool2 = F.max_pool2d(conv3, kernel_size=2)
        conv4 = F.leaky_relu(self.conv4(pool2))
        pool3 = F.max_pool2d(conv4, kernel_size=2)
        conv5 = F.leaky_relu(self.conv5(pool3))
        conv6 = F.leaky_relu(self.conv6(conv5))

        up7 = F.interpolate(conv6, scale_factor=2, mode="nearest")
        up7 = self.conv7(up7)
        up7 = torch.cat([up7, conv4], dim=1)
        conv7 = F.leaky_relu(self.conv7(up7))

        conv8 = self.conv8(conv7)

        out = conv8

        return out

class GlobalEncoder(nn.Module):
    color_channels = 3

    def __init__(self, img_size, name='resnet18', **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        pretrained = kwargs.pop('pretrained', False)
        n_features = kwargs.pop('n_features', 128)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                   resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            if self.with_pool:
                size = self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
                seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size))
            self.encoder = nn.Sequential(*seq)

        out_ch = get_output_size(self.color_channels, img_size, self.encoder)
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features
        self.out_ch = out_ch
        self.fc = fc

    def forward(self, x):
        return self.fc(self.encoder(x).flatten(1))