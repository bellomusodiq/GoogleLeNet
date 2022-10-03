import torch
import torch.nn as nn

from .layers import AuxOutputLayer, InceptionLayer


class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(192)
        self.inception_3a = InceptionLayer(
            input_channel=192,
            conv_1_out=64,
            conv_3_out=128,
            conv_3_reduce_out=96,
            conv_5_out=32,
            conv_5_reduce_out=16,
            pool_proj=32,
            downsampling=True,
        )
        self.inception_3d = InceptionLayer(
            input_channel=256,
            conv_1_out=128,
            conv_3_out=192,
            conv_3_reduce_out=128,
            conv_5_out=96,
            conv_5_reduce_out=32,
            pool_proj=64,
        )
        self.inception_4a = InceptionLayer(
            input_channel=480,
            conv_1_out=192,
            conv_3_out=208,
            conv_3_reduce_out=96,
            conv_5_out=48,
            conv_5_reduce_out=16,
            pool_proj=64,
            downsampling=True,
        )
        self.inception_4b = InceptionLayer(
            input_channel=512,
            conv_1_out=160,
            conv_3_out=224,
            conv_3_reduce_out=112,
            conv_5_out=64,
            conv_5_reduce_out=24,
            pool_proj=64,
        )
        self.aux_output_layer1 = AuxOutputLayer(
            input_size=512, pool_stride=3, conv_1_out=128, fc1_out=1024
        )
        self.inception_4c = InceptionLayer(
            input_channel=512,
            conv_1_out=128,
            conv_3_out=256,
            conv_3_reduce_out=128,
            conv_5_out=64,
            conv_5_reduce_out=24,
            pool_proj=64,
        )
        self.inception_4d = InceptionLayer(
            input_channel=512,
            conv_1_out=112,
            conv_3_reduce_out=144,
            conv_3_out=288,
            conv_5_reduce_out=32,
            conv_5_out=64,
            pool_proj=64,
        )
        self.inception_4e = InceptionLayer(
            input_channel=528,
            conv_1_out=256,
            conv_3_reduce_out=160,
            conv_3_out=320,
            conv_5_reduce_out=32,
            conv_5_out=128,
            pool_proj=128,
        )
        self.aux_output_layer2 = AuxOutputLayer(
            input_size=528, pool_stride=3, conv_1_out=128, fc1_out=1024
        )
        self.inception_5a = InceptionLayer(
            input_channel=832,
            conv_1_out=256,
            conv_3_reduce_out=160,
            conv_3_out=320,
            conv_5_reduce_out=32,
            conv_5_out=128,
            pool_proj=128,
            downsampling=True,
        )
        self.inception_5b = InceptionLayer(
            input_channel=832,
            conv_1_out=384,
            conv_3_reduce_out=192,
            conv_3_out=384,
            conv_5_reduce_out=48,
            conv_5_out=128,
            pool_proj=128,
        )
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.4)
        self.output_layer = nn.Linear(1024, output_layer)

    def forward(self, x):
        aux_out1 = None
        aux_out2 = None
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.inception_3a(x)
        x = self.inception_3d(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        if self.training:
            aux_out1 = self.aux_output_layer1(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.training:
            aux_out2 = self.aux_output_layer2(x)
        x = self.inception_4e(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x, aux_out1, aux_out2
