import torch
import torch.nn as nn
import torch.nn.functional as F

# Input = 1 Image, Output = 2(Tumor or Normal MRI)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            # Input Image Shape: (1, 224, 224) GrayScale
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            # 4 Predictions: Normal, Glioma, Meningioma, Pituitary
            nn.Linear(in_features=84, out_features=4)
        )

    def forward(self, X):
        X = self.cnn(X)
        X = X.view(X.size(0), -1)
        X = self.fc_layer(X)
        return X


# Model Formats:

# UNET:
# Designed for biomedical imaging with small datasets

# YOLOv8(YOU ONLY LOOK ONCE)
# Strong object detection

