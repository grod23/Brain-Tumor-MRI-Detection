import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Image Shape: (Batch Size, Channels, Height, Width) = (32, 1, 224, 224) Gray Scale.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            # nn.BatchNorm2d(6),
            # Could try nn.Silu() or nn.Relu()
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=5),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            # nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=5)
        )

        # Consider implementing CBAM (Convolutional Block Attention Module)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            # nn.BatchNorm2d(246),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=120),
            # nn.BatchNorm2d(120),
            nn.Tanh(),
            # 4 Predictions: Normal, Glioma, Meningioma, Pituitary.
            nn.Linear(in_features=120, out_features=4)
        )

    def forward(self, X):
        X = self.cnn(X)
        # X.size(0) grabs first value of size which is batch size or how many images.
        X = X.view(X.size(0), -1)
        X = self.fc_layer(X)
        return X

# Model Formats:

# UNET:
# Designed for biomedical imaging with small datasets

# YOLOv8(YOU ONLY LOOK ONCE)
# Strong object detection

