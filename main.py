import numpy as np
import kagglehub
from PIL import Image
from model import Model
# Neural Network Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW, lr_scheduler

# Computer Vision Libraries
import cv2
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# Graphing Library
import matplotlib.pyplot as plt

# Evaluation Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from dataset import MRI


# Kaggle Brain MRI Tumor Dataset

# Crystal Clean Version: No Duplicates, Proper Labels, and Consistent Size
# https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset

# 18606 Images
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

# Pre-processing:
# Resizing
# Intensity Normalization
# Grayscale Conversion
# Noise Reduction
# Skull Stripping


# Original Dataset has these:
# Histogram Equalization
# Data Augmentation


def main():
    # print("CUDA Available:", torch.cuda.is_available())
    # print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"

    # 7 images per patient. First image is original while the rest are augmented, so we will only grab the original
    # Every 7 images is the original

    mri_images = MRI()
    X = mri_images.images
    y = mri_images.labels

    # Brain MRI Images Visualization
    random_index = np.random.choice(X.shape[0], 6, replace=False)  # Choose 6 Random Indexes for MRI Images
    # Random Mri Images
    mri_images = [X[i] for i in random_index]
    plt.figure(figsize=(10, 5))
    print(f'Random Index: {random_index}')
    # Show Images
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(mri_images[i].permute(1, 2, 0))
        plt.title(f"Label: {int(y[random_index[i]])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Set: Optimizer, Loss Function, Activation Function, Train Test Validation Split
    # Set: Epochs, Learning Rate, Batches
    # Possibly Set: Weight Decay, Dropout Probability
    model = Model()
    epochs = 1000
    batches = 10
    learning_rate = 0.001
    weight_decay = 1e-4
    loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # End Training Early if no longer decreasing loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    # Pre-processing

    # Create Tensor Datasets
    # train_dataset = TensorDataset(X_train, y_train)

    # Create Tensor DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True)  # Only want shuffle when training
    # Train


if __name__ == '__main__':
    main()
