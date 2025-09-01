import torch
import torch.nn as nn
import torch.nn.functional as F

# Input = 1 Image, Output = 2(Tumor or Normal MRI)
class Model(nn.Module):
    def __init__(self, input_features=1, input1=16, output=2):
        super(Model, self).__init__()
        self.input_features = input_features
        self.input1 = input1
        self.output = output
        self.conv_kernel = 3
        self.conv_stride = 1
        self.pooling_kernel = 2
        self.pooling_stride = 2

        self.conv1 = nn.Conv2d(self.input_features, input1, self.conv_kernel, self.pooling_stride)
        self.conv2 = nn.Conv2d(self.input1, output, self.conv_kernel, self.pooling_stride)
        self.fc1 = nn.Linear(self.input_features, self.input1)
        self.fc2 = nn.Linear(self.input1, output)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, self.pooling_kernel, self.pooling_stride)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, self.pooling_kernel, self.pooling_stride)
        # Flatten
        # X.view()



