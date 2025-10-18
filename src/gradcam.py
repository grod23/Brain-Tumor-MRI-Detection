import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
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
        loss = prediction[0, target_class]
        loss.backward()

        # Compute weights: global average pooling of gradients
        weights = F.adaptive_avg_pool2d(self.gradients, 1)  # shape: [B, C, 1, 1]
        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        # Remove all negative values
        cam = F.relu(cam)
        # Upsample to match original image shape
        cam = F.interpolate(cam, image_shape, mode="bilinear", align_corners=False) # Shape: (1, 1, 224, 224)

        # Min Max Normalization
        B, C, H, W = cam.shape
        cam = cam.view(B, -1)
        cam -= cam.min(dim=1, keepdim=True)[0]
        cam /= cam.max(dim=1, keepdim=True)[0]
        cam = cam.view(B, C, H, W)
        cam = cam.squeeze().cpu().detach().numpy()  # Shape: [H, W]
        cam = np.uint8(cam * 255)  # Shape: [H, W]

        plt.figure(figsize=(10, 10))
        plt.imshow(cam)
        plt.title(f'GradCAM, Prediction: {target_class}')
        plt.show()

        self.heat_map = cam # Shape: (224, 224)

    def heatmap_overlay(self, image, target_class, alpha=0.4):
        self.generate(image, target_class)
        # Image and Heatmap must be numpy array of shape [224, 224, 3]
        image = (image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

        # Original image
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Original MRI Image")
        plt.axis('off')
        print(f'Image Shape: {image.shape}')
        print(f'Heatmap Shape: {self.heat_map.shape}')

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Original image
        plt.subplot(1, 4, 2)
        plt.imshow(image)
        plt.title("Original MRI Image")
        plt.axis('off')

        self.heat_map = cv2.applyColorMap(self.heat_map, cv2.COLORMAP_JET)

        # Heat Map image
        plt.subplot(1, 4, 3)
        plt.imshow(self.heat_map)
        plt.title("Heat MRI Image")
        plt.axis('off')

        overlay_image = cv2.addWeighted(image, alpha, self.heat_map, 1 - alpha, 0)

        # Overlay image
        plt.subplot(1, 4, 4)
        plt.imshow(overlay_image)
        plt.title("Overlay MRI Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return overlay_image




