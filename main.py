import numpy as np
import kagglehub
from PIL import Image
import glob
import os
import re
import sys
from sklearn.model_selection import train_test_split

from model import Model
# Neural Network Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

# Computer Vision Libraries
import cv2
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# Graphing Library
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Evaluation Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from dataset import MRI, collect_image_paths, get_data_split


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

# To Identify Tumor:
# Loss Function: Binary Cross Entropy
# Inputs: 1 Image
# Outputs: 2- Normal or Tumor

# To Identify Type of Tumor:
# Loss Function: Cross Entropy
# Inputs: 7 Images
# Outputs: 4 - Normal, Glioma, Meningioma, Pituitary

def main():
    print(f'Device Available: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model().to(device)

    epochs = 2
    batches = 32
    learning_rate = 0.001
    weight_decay = 1e-4
    loss_fn = nn.CrossEntropyLoss()
    # Couples with Weight Decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # End Training Early if no longer decreasing loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Get Train Val Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = get_data_split()

    # Create DataSet Instances
    train_dataset = MRI(X_train, y_train)
    validation_dataset = MRI(X_val, y_val)
    test_dataset = MRI(X_test, y_test)


    # Create DataLoaders
    training_loader = DataLoader(train_dataset, batch_size=batches, num_workers=4, shuffle=True) # Only Shuffle Training Data
    validation_loader = DataLoader(validation_dataset, batch_size=batches, num_workers=4, shuffle=False)
    testing_loader = DataLoader(test_dataset, batch_size=batches, num_workers=4, shuffle=False)
    # Num_workers specify how many parallel subprocesses are used to load the data
    # DataLoaders also add Batch Size to Shape: (batch_size, 1, 224, 224)

    # Loss Tracking
    loss_track = []
    # Validation Tracking
    val_track = []
    # Accuracy
    correct = 0
    total = 0
    # Training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in training_loader:
            # Use GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Reset Gradients
            optimizer.zero_grad()
            # Get y_hat
            y_predicted = model(X_batch)
            # print(f'Prediction: {y_predicted.argmax(dim=1)}')
            # print(f'Label: {y_batch}')
            # Get Loss
            loss = loss_fn(y_predicted, y_batch) # y_batch must be of type LongTensor()
            # Backpropagation
            loss.backward()
            # Update Learnable Parameters
            optimizer.step()
            # Update Epoch Loss
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(training_loader)
        loss_track.append(train_loss)
        print(f'Training Epoch: {epoch}, Loss: {train_loss}')

        # Validation for the current epoch
        model.eval()
        # No Gradient Calculation
        with torch.no_grad():
            batch_loss = 0
            for X_val, y_val in validation_loader:
                # Use GPU
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_predicted = model(X_val)
                loss = loss_fn(y_predicted, y_val)
                batch_loss += loss.item()
                correct += (y_predicted.argmax(dim=1) == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = batch_loss / len(validation_loader)
        val_track.append(avg_val_loss)
        print(f'Training Epoch: {epoch}, Validation Loss: {avg_val_loss}')

    print(f'Correct: {correct}, Total Images: {total}')
    print(f'Validation Accuracy: {correct / total}')

    # Visualize Training and Validation Loss
    plt.figure(figsize=(15, 9))
    plt.plot(loss_track, c='b', label='Train Loss')
    plt.plot(val_track, c='r', label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.show()


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
