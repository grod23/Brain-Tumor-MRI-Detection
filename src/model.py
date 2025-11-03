import torch.nn as nn
from cbam import CBAM

# Highest Validation Accuracy 89%
class Model(nn.Module):
    def __init__(self, dropout_probability):
        super(Model, self).__init__()
        # Dropout
        self.dropout_probability = dropout_probability
        # Image Shape: (Batch Size, Channels, Height, Width) = (16, 1, 224, 224) Gray Scale.
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            #nn.SiLU(),
            nn.ReLU(),
            CBAM(channels=16, reduction=4),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAM(channels=32, reduction=8),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(channels=64, reduction=8),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.global_avg = nn.AdaptiveAvgPool2d((3, 3))  # Output: (B, 3 * 3 * 64, 1, 1)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=576, out_features=256),
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
        X = self.global_avg(X)
        # X.size(0) grabs first value of size which is batch size/amount of images.
        X = X.view(X.size(0), -1)
        X = self.fc_layer(X)
        return X


# Model Formats:

# UNET:
# Designed for biomedical imaging with small datasets

# YOLOv8(YOU ONLY LOOK ONCE)
# Strong object detection

