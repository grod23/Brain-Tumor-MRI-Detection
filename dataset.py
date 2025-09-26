# File Handling
import glob
import os.path
import re

import cv2
import torch
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit
from torch.optim.radam import radam
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import sys
from collections import defaultdict


# Download latest version
# path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")

# Normal Image Size is 224x224
# Image Size Varies
# 7 images per patient. First image is original while the rest are augmented
# Every 7 images is the original
def collect_image_paths():
    image_paths = []
    labels = []
    groups = []

    # Gets the File Number in order to sort numerically
    def extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group())

    def get_images(path, label):
        # Image Paths for Specific Patient
        patients = defaultdict(list)

        # Sort Files Numerically
        files = glob.iglob(path)
        files = sorted(files, key=extract_number)

        # Loop and Categorize Images by Patient
        for file in files:
            file_name = os.path.basename(file).removesuffix('.jpg')
            parts = file_name.split('_')
            base_name = '_'.join(parts[0:2])
            patients[base_name].append(file)


        for patient_id, images in patients.items():
            if len(images) != 7:
                raise Exception('Not 7 Images')
            for image in images:
                # Append Image Features(image_path, label, and id)
                image_paths.append(image)
                labels.append(label)
                groups.append(patient_id)

    # Load Image Paths
    get_images("./Brain_Tumor_Dataset/Normal/*.jpg", 0)
    get_images("./Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg", 1)
    get_images("./Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg", 2)
    get_images("./Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg", 3)

    return np.array(image_paths), np.array(labels), np.array(groups)

def get_data_split():
    image_paths, labels, groups = collect_image_paths()

    print(f'Total Images: {len(image_paths)}')
    print(f'Total Labels: {len(labels)}')
    print(f'Total Groups: {len(np.unique(groups))}')

    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_index, test_index = next(gss.split(image_paths, labels, groups))

    X_train = image_paths[train_index]
    y_train = labels[train_index]
    X_testing = []
    y_testing = []

    # Remove Augmented Images from Testing
    for index in test_index:
        file_name = os.path.basename(image_paths[index]).removesuffix('.jpg').split('_')
        if len(file_name) == 2:
            # It is original
            X_testing.append(image_paths[index])
            y_testing.append(labels[index])

    print(f'X Train: {len(X_train)}')
    print(f'y Train: {len(y_train)}')
    print(f'X Test: {len(X_testing)}')
    print(f'Y test: {len(y_testing)}')

    X_val, X_test, y_val, y_test = train_test_split(X_testing, y_testing, test_size=0.5, random_state=42, stratify=y_testing)

    # Optional: check for group leakage
    assert set(groups[train_index]).isdisjoint(set(groups[test_index])), "Group leakage detected!"
    print(f'Train size: {len(X_train)} images from {len(np.unique(groups[train_index]))} patients')
    print(f'Test size: {len(X_testing)} images from {len(np.unique(groups[test_index]))} patients')

    # Normal: 0
    # Glioma: 1
    # Meningioma: 2
    # Pituitary: 3

    return np.array(X_train), torch.LongTensor(y_train), np.array(X_val), torch.LongTensor(y_val), np.array(
        X_testing), torch.LongTensor(y_testing)

def compute(image_paths):
    pixel_values = []

    for path in image_paths:
        # Read grayscale image as float32 in range [0, 1]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        pixel_values.append(img.flatten())

    # Concatenate all pixels from all images
    all_pixels = np.concatenate(pixel_values)

    mean = np.mean(all_pixels)
    std = np.std(all_pixels)

    return mean, std



class MRI(Dataset):
    def __init__(self, image_paths, labels, testing=False):
        self.image_paths = image_paths
        self.labels = labels
        # Resize, Adjust Tensor Shape, Min-Max Normalization, Z-Score Normalization
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.1891], std=[0.1936])
                                             ])
        self.testing = testing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        #if not self.testing:
            #print(f'Training Image Path: {image_path}')

        label = self.labels[index]

        # Load Image on Demand
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Preprocessing

        # Histogram Equalization
        image = cv2.equalizeHist(image)

        if self.testing:
            #print(f'Validation Image Path: {image_path}')
            # image = image.numpy()

            # Noise Reduction:
            # For Salt and Pepper Noise
            image = cv2.medianBlur(image, 3)  # Kernel Size must be odd

            # Bilateral Filter to smooth preserve edges very well
            image = cv2.bilateralFilter(image, d=3, sigmaColor=10, sigmaSpace=10)

            # For testing:
            # No augmented images

        # Normalize and Resize
        image = self.transform(image)

        return image, label

    # Data Augmentation already included in DataSet:

    # Salt and Pepper Noise: Introducing random noise by setting pixels to white or black based on a specified intensity.
    # Histogram Equalization: Applying histogram equalization to enhance the contrast and details in the images.
    # Rotation: Rotating the images clockwise or counterclockwise by a specified angle.
    # Brightness Adjustment: Modifying the brightness of the images by adding or subtracting intensity values.
    # Horizontal and Vertical Flipping: Flipping the images horizontally or vertically to create mirror images.

    # Image Preprocessing Notes:

    # For Robustness: Data Augmentation Examples Above
    # Data Augmentation should be used for TRAINING only. Using augmented images in validation and testing sets may
    # cause overfitting and unreliable accuracy.

    # Standardization(Z-Score Normalization): (X - mean(X)/ SD(X)): - Centers data around 0 with unit variance
    # Improves convergence. Without, optimization may have unwanted oscillations during convergence. Required for MRI
    # Preferred when outliers are present
    # transforms.Normalize(mean=[], std=[])

    # Min-Max Normalization: Scales data point values to be between [0,1]. Typically, not required for MRI, Z-Score
    # Normalization would be more important
    # transforms.ToTensor()

    # Histogram: Visual representation of quantitative data distribution
    # Histogram Equalization adjusts the contrast of an image by adjusting histogram to be uniform throughout.

    # Noise Reduction: Removes distortions, grains, speckles while keeping
    # important detail and clarity(Use for testing only)

    # Skull Stripping: Strong potential only if every image model sees is in this format. For deployment,
    # every image must also be preprocessed with Skull Stripping.

    # Intensity Normalization: MRI intensities are not standardized. The same tissue can appear with a different
    # brightness. Intensity Normalization brings the intensities to a common scale in order to separate tissue and
    # segment the brain. For deployment, every image must also be preprocessed with Intensity Normalization.