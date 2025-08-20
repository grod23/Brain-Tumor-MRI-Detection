import numpy as np
import glob
import os
import re
# Neural Network Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# Computer Vision Libraries
import cv2
from torchvision import datasets, transforms
from torchvision.utils import make_grid
# Graphing Library
import matplotlib.pyplot as plt
# Evaluation Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
def main():

    # 7 images per patient. First image is original while the rest are augmented, so we will only grab the original
    # Every 7 images is the original

    # Normal Image Size is 224x224
    # Image Size Varies

    def load_images(path):
        images = []
        for file in (glob.iglob(path)):
            filename = os.path.splitext(os.path.basename(file))[0]  # e.g., 'N_1' or 'N_1_BR'
            if re.fullmatch(r'[A-Z]_\d+', filename):  # e.g., 'N_1', 'G_2', etc.
                image = cv2.imread(file)
                # Ensure Consistent Image Size
                image = cv2.resize(image, (224, 224))
                # Ensure Consistent RGB Order
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

        return np.array(images)

    # Load Image Arrays
    normal = load_images("./Brain_Tumor_Dataset/Normal/*.jpg")
    glioma = load_images("./Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg")
    meningioma = load_images("./Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg")
    pituitary = load_images("./Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg")

    # Create Image Tensors
    normal = torch.tensor(normal, dtype=torch.uint8)
    glioma = torch.tensor(glioma, dtype=torch.uint8)
    meningioma = torch.tensor(meningioma, dtype=torch.uint8)
    pituitary = torch.tensor(pituitary, dtype=torch.uint8)
    # Combine Tumor Tensors
    tumors = torch.cat((glioma, meningioma, pituitary), dim=0)

    # Array Shapes: (Number of Images, 224, 224, 3)
    # (Number of Images, Width, Height, Color Channels)
    # (18606, 224, 224, 3)

    print(tumors.shape)
    print(normal.shape)

    # Assign 1 for tumors, 0 for normal
    y_tumor = torch.ones(len(tumors))
    y_normal = torch.zeros(len(normal))

    # Combine Data
    X = torch.cat((normal, tumors), dim=0)
    y = torch.cat((y_normal, y_tumor), dim=0)

    print(f"Image tensor shape: {X.shape}")
    print(f"Label tensor shape: {y.shape}")

    # Brain MRI Images Visualization
    random_index = np.random.choice(X.shape[0], 6, replace=False)
    mri_images = [X[i] for i in random_index]
    plt.figure(figsize=(10, 5))
    print(random_index)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(mri_images[i])
        plt.title(f"Label: {int(y[random_index[i]])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
