import torch
import torch.nn as nn
# Gaussian Kernel
from scipy.stats import multivariate_normal
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CBAM(nn.Module):
    def __init__(self, channels, reduction):
        super(CBAM, self).__init__()
        # Says which feature map is important and refines it.
        self.channel_gate = Channel_Gate(channels, reduction)
        # Mask that highlights the features of the tumor. Conveys what within the feature map is essential to learn.
        self.spatial_gate = Spatial_Gate()


    def forward(self, X):
        X = self.channel_gate(X)
        X = self.spatial_gate(X)
        return X


class Channel_Gate(nn.Module):
    # Reductions used to reduce feature maps
    def __init__(self, gate_channels, reduction):
        super(Channel_Gate, self).__init__()
        # Squeeze Excitation Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction, gate_channels, bias=False)

        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # Need to be Shape: (Batch, Channels)
        X1 = self.avg_pool(X)
        X1 = self.mlp(X1)
        X2 = self.max_pool(X)
        X2 = self.mlp(X2)
        channel_attention_sum = X1 + X2
        channel_attention_sum = self.sigmoid(channel_attention_sum).view(X.size(0), X.size(1), 1, 1)  # Shape: (Batch, Channels, 1, 1)
        refined_channel_attention = X * channel_attention_sum
        return refined_channel_attention

class Spatial_Gate(nn.Module):
    def __init__(self):
        super(Spatial_Gate, self).__init__()
        # Gaussian Center Weight Map
        self.center_weight = gaussian_weight_map(size=224, sigma=0.1).to(device)
        self.bias_strength = 8.0
        self.kernel_size = 7
        self.compress = Channel_Pool()
        self.spatial = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=self.kernel_size, padding=(self.kernel_size-1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # Flatten + Reduction
        X_compress = self.compress(X)
        # Conv Layer
        X_compress = self.spatial(X_compress)
        # Center Bias
        # ALWAYS resize bias to match current feature map size
        current_size = X_compress.shape[2:]  # (H, W) of current feature map
        center_bias = nn.functional.interpolate(self.center_weight, size=current_size,
                                         mode='bilinear', align_corners=False)
        # Add center bias to gradients
        X_compress = (self.bias_strength * center_bias) + X_compress
        X_compress = self.sigmoid(X_compress)
        refined_X = X * X_compress

        return refined_X

class Channel_Pool(nn.Module):
    def forward(self, X):
        return torch.cat( (torch.max(X,1)[0].unsqueeze(1), torch.mean(X,1).unsqueeze(1)), dim=1)


def gaussian_weight_map(size=224, sigma=0.1):
    # Grid (Tuple of N-dimensional arrays corresponding to coordinate vectors).
    X, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # Stacks position into 3D array/
    position = np.dstack((X, y))
    # Mean and Covariance
    mean = [0, 0] # Centered at (0, 0)
    # Covariance matrix controls spread
    # [[sigma^2, 0], [0, sigma^2]] for circular Gaussian
    covariance = [[sigma ** 2, 0],
                  [0, sigma ** 2]]
    rv = multivariate_normal(mean, covariance)
    heat_kernel = rv.pdf(position)

    # Make it learnable (optional - can be fixed too)
    heat_kernel = torch.tensor(heat_kernel, dtype=torch.float32)

    # heat_kernel = nn.Parameter(heat_kernel.unsqueeze(0).unsqueeze(0))
    heat_kernel = heat_kernel.unsqueeze(0).unsqueeze(0)
    # Normalize to [0,1]
    kernel_min = heat_kernel.min()
    kernel_max = heat_kernel.max()
    heat_kernel = (heat_kernel - kernel_min) / (kernel_max - kernel_min)
    return heat_kernel


