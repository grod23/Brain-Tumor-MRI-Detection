import torch.nn as nn
import sys
from cbam import CBAM
class Model(nn.Module):
    def __init__(self, dropout_probability):
        super(Model, self).__init__()
        # Dropout
        self.dropout_probability = dropout_probability
        # Image Shape: (Batch Size, Channels, Height, Width) = (B, 1, 224, 224)
        self.cnn = nn.Sequential(
            # Block 1: 1 → 16 channels
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112

            # Block 2: 16 → 32 channels
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAM(channels=32, reduction=8),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 → 56

            # Block 3: 32 → 64 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(channels=64, reduction=8),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 → 28

            # Block 4: 64 → 64 channels (deeper features)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAM(channels=128, reduction=8),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 → 14
        )
        # After CNN: (B, 64, 14, 14)

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 64, 1, 1)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout_probability),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # 4 Predictions: Normal, Glioma, Meningioma, Pituitary.
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.global_avg(X)
        # X.size(0) grabs first value of size which is batch size or how many images.
        X = X.view(X.size(0), -1)
        X = self.fc_layer(X)
        return X

# Model Formats:

# UNET:
# Designed for biomedical imaging with small datasets

# YOLOv8(YOU ONLY LOOK ONCE)
# Strong object detection

