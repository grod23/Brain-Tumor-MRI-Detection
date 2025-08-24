import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.functional):
    def __init__(self, input_features=1, input1=16, output=2, conv_kernel=3, conv_stride=1, pooling_kernel=2, pooling_stride=2):
        super(Model, self).__init__()
        self.input_features = input_features
        self.input1 = input1
        self.output = output
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride

        self.conv1 = nn.Conv2d(self.input_features, input1, conv_kernel, pooling_stride)
        self.conv2 = nn.Conv2d(self.input1, output, conv_kernel, pooling_stride)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, self.pooling_kernel, self.pooling_stride)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, self.pooling_kernel, self.pooling_stride)
        # Flatten



