import torch.nn as nn
import sys
from cbam import CBAM
class Model(nn.Module):
    def __init__(self, dropout_probability):
        super(Model, self).__init__()
        # Dropout
        self.dropout_probability = dropout_probability
        # Image Shape: (Batch Size, Channels, Height, Width) = (8, 1, 224, 224) Gray Scale.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=5),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Convolutional Block Attention Module
            CBAM(channels=16, reduction=4),
            nn.MaxPool2d(kernel_size=3, stride=5)
        )


        self.global_avg = nn.AdaptiveAvgPool2d((1, 1)) # Shape: (B, 16, 1, 1)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout_probability),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=120),
            nn.BatchNorm1d(120),
            nn.Dropout(self.dropout_probability),
            nn.ReLU(),
            # 4 Predictions: Normal, Glioma, Meningioma, Pituitary.
            nn.Linear(in_features=120, out_features=4)
        )

    def forward(self, X):
        X = self.cnn(X)
        # X.size(0) grabs first value of size which is batch size or how many images.
        X = X.view(X.size(0), -1)
        # print(f'Output Shape: {X.shape}')
        X = self.fc_layer(X)
        return X

# Model Formats:

# UNET:
# Designed for biomedical imaging with small datasets

# YOLOv8(YOU ONLY LOOK ONCE)
# Strong object detection

