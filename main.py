import numpy as np
import glob
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

# 3264 Images
# Normal: 3066
# Glioma: 6307
# Meningioma: 6391
# Pituitary: 5908

# image Shape: (224, 224, 3) 224x224, 3 Color Channels
def main():
    # Import MRI Images

    normal = []
    glioma_tumor = []
    meningioma_tumor = []
    pituitary_tumor = []

    path = "./Brain_Tumor_Dataset/Normal/*.jpg"
    # Normal Image Size is 224x224
    for file in glob.iglob(path):
        image = cv2.imread(file)
        # Ensures Color Channel order is RGB
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        normal.append(image)

    # Image Size Varies
    path = "./Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg"
    for file in glob.iglob(path):
        image = cv2.imread(file)
        # Ensures Consistent Image Size
        image = cv2.resize(image, (224, 224))
        # Ensures Color Channel order is RGB
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        glioma_tumor.append(image)

    path = "./Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg"
    for file in glob.iglob(path):
        image = cv2.imread(file)
        # Ensures Consistent Image Size
        image = cv2.resize(image, (224, 224))
        # Ensures Color Channel order is RGB
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        meningioma_tumor.append(image)

    pituitary_path = "./Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg"
    for file in glob.iglob(pituitary_path):
        image = cv2.imread(file)
        # Ensures Consistent Image Size
        image = cv2.resize(image, (224, 224))
        # Ensures Color Channel order is RGB
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        pituitary_tumor.append(image)

    for img in glioma_tumor:
        print(img.shape)
    #  Convert Lists into Numpy Arrays
    normal = np.array(normal)
    glioma = np.array(glioma_tumor)
    meningioma = np.array(meningioma_tumor)
    pituitary = np.array(pituitary_tumor)

    # Tensor Shapes: (Number of Images, 224, 224, 3)


if __name__ == '__main__':
    main()
