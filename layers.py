import torch
import torch.nn as nn


class InceptionLayer(nn.Module):
    def __init__(
        self,
        input_channel,
        conv_1_out,
        conv_3_out,
        conv_3_reduce_out,
        conv_5_out,
        conv_5_reduce_out,
        pool_proj,
        downsampling=False,
    ):
        super(InceptionLayer, self).__init__()
        self.downsampling = downsampling
        self.input_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_1 = nn.Conv2d(input_channel, conv_1_out, kernel_size=1)
        self.batchnorm_1 = nn.BatchNorm2d(conv_1_out)
        self.relu = nn.ReLU()
        self.conv_3_reduce = nn.Conv2d(input_channel, conv_3_reduce_out, kernel_size=1)
        self.batchnorm_3_reduce = nn.BatchNorm2d(conv_3_reduce_out)
        self.conv_3 = nn.Conv2d(conv_3_reduce_out, conv_3_out, kernel_size=3, padding=1)
        self.batchnorm_3 = nn.BatchNorm2d(conv_3_out)
        self.conv_5_reduce = nn.Conv2d(input_channel, conv_5_reduce_out, kernel_size=1)
        self.batchnorm_5_reduce = nn.BatchNorm2d(conv_5_reduce_out)
        self.conv_5 = nn.Conv2d(conv_5_reduce_out, conv_5_out, kernel_size=5, padding=2)
        self.batchnorm_5 = nn.BatchNorm2d(conv_5_out)
        self.maxpool = nn.MaxPool2d(3, 1, padding=1)
        self.conv_maxpool = nn.Conv2d(input_channel, pool_proj, kernel_size=1)
        self.batchnorm_maxpool = nn.BatchNorm2d(pool_proj)

    def forward(self, input):
        if self.downsampling:
            input = self.input_maxpool(input)
        x_1 = self.relu(self.batchnorm_1(self.conv_1(input)))
        x_3 = self.relu(self.batchnorm_3_reduce(self.conv_3_reduce(input)))
        x_3 = self.relu(self.batchnorm_3(self.conv_3(x_3)))
        x_5 = self.relu(self.batchnorm_5_reduce(self.conv_5_reduce(input)))
        x_5 = self.relu(self.batchnorm_5(self.conv_5(x_5)))
        pool_out = self.maxpool(input)
        pool_out = self.conv_maxpool(pool_out)
        output = torch.cat((x_1, x_3, x_5, pool_out), dim=1)
        return output


class AuxOutputLayer(nn.Module):
    def __init__(
        self,
        input_size,
        pool_stride,
        conv_1_out,
        fc1_out,
        output_size=257,
    ):
        super(AuxOutputLayer, self).__init__()
        self.conv_1_out = conv_1_out
        self.avg_pool = nn.AvgPool2d(5, stride=pool_stride)
        self.conv = nn.Conv2d(input_size, conv_1_out, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(conv_1_out)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * conv_1_out, fc1_out)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(fc1_out, output_size)

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = x.view(-1, 16 * self.conv_1_out)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x
