import matplotlib.pyplot as plt

from train import train
from model import Model
from gradcam import GradCAM
from dataset import MRI, get_data_split
import torch
import numpy as np
import cv2

# Kaggle Brain MRI Tumor Dataset

# Crystal Clean Version: No Duplicates, Proper Labels, and Consistent Size
# https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset

# 21672 Total Images
# 18606 Tumor Images
# Normal: 3066
# Glioma: 6307
# Meningioma: 6391
# Pituitary: 5908

# After Image Processing
# Tumor: 2568
# Normal: 438
# image Shape: (224, 224, 3) 224x224, 3 Color Channels

# Activation Function: ReLU
# Optimizer: ADAM

# To Identify Type of Tumor:
# Loss Function: Cross Entropy
# Inputs: 7 Images
# Outputs: 4 - Normal, Glioma, Meningioma, Pituitary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparameters
    epochs = 30
    batch_size = 8
    learning_rate = 0.00001
    weight_decay = 0.001
    dropout_probability = 0.3
    model = Model(dropout_probability).to(device)

    # print(model)
    train(epochs, batch_size, learning_rate, weight_decay, model)

    _, _, _, _, X_test, y_test = get_data_split()
    mri = MRI(X_test, y_test)
    image, label = mri.__getitem__(640)
    image = image.to(device)
    cam = GradCAM(model, target_layer_name='cnn.4')
    heatmap_image = cam.heatmap_overlay(image, target_class=1)
    # Convert image tensor to numpy for visualization
    image_np = np.uint8(image.detach().cpu().numpy().squeeze())  # Shape: (224, 224)

    # Display side by side
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title(f"Original MRI Image: {label}")
    plt.axis('off')

    # Grad-CAM overlay
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_image)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Test Accuracy: 0.8603712671509282
    #               precision    recall  f1-score   support
    #
    #            0       0.85      0.85      0.85       164
    #            1       0.87      0.83      0.84       373
    #            2       0.86      0.78      0.82       371
    #            3       0.86      0.99      0.92       331
    #
    #     accuracy                           0.86      1239
    #    macro avg       0.86      0.86      0.86      1239
    # weighted avg       0.86      0.86      0.86      1239

    # Visualizing Feature Maps
    # num_layers = 0
    # conv_layers = []
    # # List of the 2 sequential objects in model (nn.Sequential)
    # model_children = list(model.children())
    #
    # for child in model_children:
    #     # Checks for type Sequential
    #     if type(child) == nn.Sequential:
    #         # Want to visualize the child of model_children
    #         for layer in child.children():
    #             # If it's a Convolutional Layer
    #             if not type(layer) == nn.Linear:
    #                 # Record This Layer
    #                 num_layers += 1
    #                 conv_layers.append(layer)
    #
    # print(conv_layers)
    # print(f'X Shape: {X_train.shape}')
    # dataset = MRI(X_train, y_train, testing=False)
    # image, label = dataset.__getitem__(3233)
    # print(f'Image Shape Before: {image.shape}')
    # print(label)
    # print(image.unsqueeze(0).shape)
    # plt.imshow(image.view(image.shape[2], image.shape[1], image.shape[0]))
    # plt.show()
    # image = image.unsqueeze(0).to(device)
    # results = [conv_layers[0](image)]
    # for i in range(1, len(conv_layers)):
    #     # Input for next layer is output of last layer
    #     results.append(conv_layers[i](results[-1]))
    # print(results[0].shape)
    #
    # # Visualize
    # for layer in range(len(results)):
    #     plt.figure(figsize=(30, 10))
    #     layer_viz = results[layer].squeeze()
    #     for i, f in enumerate(layer_viz):
    #         print(f'F Shape: {f.shape}')
    #         plt.subplot(2, 8, i + 1)
    #         plt.imshow(f.detach().cpu().numpy())
    #         plt.axis('off')
    #     plt.show()

    # Brain MRI Images Visualization

    # # replace=False avoids duplicates values
    # random_index = np.random.choice(len(X), 6, replace=False)  # Choose 6 Random Indexes for MRI Images
    # plt.figure(figsize=(10, 5))
    # # print(f'Random Index: {random_index}')
    # # Show Images
    # for i in range(6):
    #     image, label = dataset[random_index[i]]
    #     image = image.view(image.shape[2], image.shape[1], image.shape[0])
    #     plt.subplot(2, 3, i + 1)
    #     # Matplotlib takes images as (Height, Width, Channel)
    #     plt.imshow(image)  # PyTorch takes images as (Channel, Height, Width)
    #     plt.title(f"Label: {int(y[random_index[i]])}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    #
    # # Histogram Plot
    # plt.figure(figsize=(10, 9))  # Wider and taller figure
    #
    # for i in range(3):
    #     image, label = dataset[random_index[i]]
    #
    #     # Convert from (C, H, W) to (H, W, C) for display
    #     image_np = image.permute(1, 2, 0).numpy()
    #
    #     # Image subplot
    #     plt.subplot(3, 2, 2 * i + 1)
    #     plt.imshow(image_np)
    #     plt.title(f"Image - Label: {int(label)}")
    #     plt.axis('off')
    #
    #     # Histogram subplot
    #     plt.subplot(3, 2, 2 * i + 2)
    #     sns.histplot(image.numpy().ravel(), kde=True, bins=50, color='skyblue')
    #     plt.title("Pixel Intensity Distribution")
    #     plt.xlabel("Pixel Value")
    #     plt.ylabel("Frequency")
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
