import torch
import torch.nn as nn

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
    def __init__(self, gate_channels, reduction=4):
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
        self.kernel_size = 7
        self.compress = Channel_Pool()
        self.spatial = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=self.kernel_size, padding=(self.kernel_size-1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X_compress = self.compress(X)
        X_compress = self.spatial(X_compress)
        X_compress = self.sigmoid(X_compress)
        refined_X = X * X_compress

        return refined_X

class Channel_Pool(nn.Module):
    def forward(self, X):
        return torch.cat( (torch.max(X,1)[0].unsqueeze(1), torch.mean(X,1).unsqueeze(1)), dim=1)
