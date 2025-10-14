import numpy as np
import torch
import torch.nn.functional as F
import cv2
import sys

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer = self.get_conv_layer(target_layer_name)
        # Placeholder for heat map
        self.heat_map = None

        # Placeholders for activations and gradients
        self.activations = None
        self.gradients = None

        # Register Hooks
        self._register_hooks()


    # Loops through models layers and returns target layer.
    def get_conv_layer(self, layer_name):
        for name, layer in self.model.named_modules():
            if layer_name == name:
                return layer
        raise Exception(f"Layer: {layer_name} Not Found")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image, target_class=None):
        # Only need Height and Width
        image_shape = (image.shape[1], image.shape[2])
        # Compute Gradients
        image.requires_grad_(True)
        prediction = self.model(image.unsqueeze(0))
        # Handle None Target Class
        if target_class is None:
            target_class = prediction.argmax(dim=1).item()

        # Reset Gradient
        self.model.zero_grad()
        loss = prediction[:, target_class]
        loss.backward()

        # Compute weights: global average pooling of gradients
        gradients = F.adaptive_avg_pool2d(self.gradients,1 )  # shape: [B, C, 1, 1]
        # Weighted sum of activations
        cam = torch.mul(gradients, self.activations).sum(dim=1, keepdim=True)
        # Remove all negative values
        cam = F.relu(cam)
        # Upsample to match original image shape
        cam = F.interpolate(cam, image_shape, mode="bilinear", align_corners=False)
        # Min Max Normalization
        B, C, H, W = cam.shape
        cam = cam.view(B, -1)
        cam -= cam.min(dim=1, keepdim=True)[0]
        cam /= cam.max(dim=1, keepdim=True)[0]
        cam = cam.view(B, C, H, W).squeeze(0).cpu().detach().numpy()
        cam = np.uint8(cam * 255)
        self.heat_map = cam # Shape: [1, H, W]

    def heatmap_overlay(self, image, target_class, alpha=0.4):
        self.generate(image, target_class)
        # Image and Heatmap must be numpy array of shape [224, 224]
        image = image.detach().cpu().numpy().squeeze(0)
        image = np.uint8(image)
        # print(f'Image Shape: {image.shape}')
        # print(f'Heatmap Shape: {self.heat_map.shape}')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.heat_map = cv2.applyColorMap(self.heat_map.squeeze(), cv2.COLORMAP_JET)
        # Turns Image and Heatmap to numpy array of shape[224, 224, 3]
        # print(f'Image Shape: {image.shape}')
        # print(f'Heatmap Shape: {self.heat_map.shape}')
        overlay_image = cv2.addWeighted(image, alpha, self.heat_map, 1 - alpha, 0)
        return overlay_image




