import numpy as np
import kagglehub
from PIL import Image
import glob
import os
import re

from sklearn.model_selection import train_test_split

from model import Model
# Neural Network Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from dataset import MRI, collect_image_paths


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
    # print("CUDA Available:", torch.cuda.is_available())
    # print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"

    model = Model()

    epochs = 1000
    batches = 10
    learning_rate = 0.001
    weight_decay = 1e-4
    loss_fn = nn.CrossEntropyLoss()
    # Couples with Weight Decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # End Training Early if no longer decreasing loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    X, y = collect_image_paths()
    # 21672 Images

    # Train Validation Test Split

    # Preserve Class Distribution using stratify
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

    # Unravel Data Array, No Longer Need Patient Array now that data is split. No risk of data leakage
    y_train = y_train.repeat_interleave(X_train.shape[1])
    y_val = y_val.repeat_interleave(X_val.shape[1])
    y_test = y_test.repeat_interleave(X_test.shape[1])

    X_train = X_train.reshape(-1)
    X_val =X_val.reshape(-1)
    X_test = X_test.reshape(-1)

    # Create DataSet Instances
    train_dataset = MRI(X_train, y_train)
    validation_dataset = MRI(X_val, y_val)
    test_dataset = MRI(X_test, y_test, testing=True)


    # Create DataLoaders
    training_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True) # Only Shuffle Training Data
    validation_loader = DataLoader(validation_dataset, batch_size=32, num_workers=4, shuffle=False)
    testing_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)
    # Num_workers specify how many parallel subprocesses are used to load the data
    # DataLoaders also add Batch Size to Shape: (32, 1, 224, 224)

    # Training

    # Loss Tracking
    loss_track = []
    # Patience Counter for Early Stopping
    patience_counter = 0

    # model.train()
    # for epoch in range(epochs):
    #     epoch_loss = 0.0
    #     best_loss = 0.0
    #     for X_batch, y_batch in training_loader:
    #         y_predicted = model(X_batch)
    #         loss = loss_fn(y_predicted, y_batch)
    #         # Reset Gradients
    #         optimizer.zero_grad()
    #         # Backpropagation
    #         loss.backward()
    #         # Update Parameters
    #         optimizer.step()
    #         # Track Epoch Loss
    #         epoch_loss += loss.item()
    #     test_loss = epoch_loss / len(training_loader)
    #     loss_track.append(test_loss)
    #     scheduler.step(test_loss)
    #
    #     if test_loss < best_loss:
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= 10:
    #             print(f'Early Stopping: {}')

    # Brain MRI Images Visualization
    y = y.repeat_interleave(X.shape[1])
    X = X.reshape(-1)
    dataset = MRI(X, y, testing=True)

    # replace=False avoids duplicates values
    random_index = np.random.choice(len(X), 6, replace=False)  # Choose 6 Random Indexes for MRI Images
    plt.figure(figsize=(10, 5))
    # print(f'Random Index: {random_index}')
    # Show Images
    for i in range(6):
        image, label = dataset[random_index[i]]
        image = image.view(image.shape[2], image.shape[1], image.shape[0])
        plt.subplot(2, 3, i + 1)
        # Matplotlib takes images as (Height, Width, Channel)
        plt.imshow(image)  # PyTorch takes images as (Channel, Height, Width)
        plt.title(f"Label: {int(y[random_index[i]])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Histogram Plot
    plt.figure(figsize=(10, 9))  # Wider and taller figure

    for i in range(3):
        image, label = dataset[random_index[i]]

        # Convert from (C, H, W) to (H, W, C) for display
        image_np = image.permute(1, 2, 0).numpy()

        # Image subplot
        plt.subplot(3, 2, 2 * i + 1)
        plt.imshow(image_np)
        plt.title(f"Image - Label: {int(label)}")
        plt.axis('off')

        # Histogram subplot
        plt.subplot(3, 2, 2 * i + 2)
        sns.histplot(image.numpy().ravel(), kde=True, bins=50, color='skyblue')
        plt.title("Pixel Intensity Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
