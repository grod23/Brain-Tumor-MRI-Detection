# File Handling
import glob
import os.path
import re

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


# Download latest version
# path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")

# Normal Image Size is 224x224
# Image Size Varies
# 7 images per patient. First image is original while the rest are augmented
# Every 7 images is the original
def collect_image_paths():
    image_paths = []
    labels = []

    # Gets the File Number in order to sort numerically
    def extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group())

    def get_images(path, label):
        # Image Paths for Specific Patient
        patient_image_paths = []
        previous_base_name = ' '

        # Sort Files Numerically
        files = glob.iglob(path)
        files = sorted(files, key=extract_number)

        # Loop and Categorize Images by Patient
        for file in files:
            file_name = os.path.basename(file).removesuffix('.jpg')
            parts = file_name.split('_')
            base_name = '_'.join(parts[0:2])
            # If Base Name Matches Previous Base Name
            if base_name == previous_base_name:
                # Add to Patient Array
                patient_image_paths.append(file)

            else:
                # Append Patient Image Paths if it's not empty
                if patient_image_paths:
                    image_paths.append(patient_image_paths)
                    labels.append(label)
                # Reset Patient Image Paths
                patient_image_paths.clear()
                # Append First File Image to New Patient Image Path List
                patient_image_paths.append(file)
                # Set New Previous Base Name
                previous_base_name = base_name

    # Load Image Paths
    get_images("./Brain_Tumor_Dataset/Normal/*.jpg", 0)
    get_images("./Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg", 1)
    get_images("./Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg", 2)
    get_images("./Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg", 3)

    # Return Image Paths and Corresponding Labels
    # print(f'Image Path Length: {len(image_paths)}')
    # print(f'Lengh of Labels: {len(labels)}')
    return np.array(image_paths), torch.tensor(labels, dtype=torch.float32)

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
        label = self.labels[index]

        # Load Image on Demand
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Preprocessing

        # Histogram Equalizationbw
        image = cv2.equalizeHist(image)

        if self.testing:
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

    # Min-Max Normalization: Scales data point values to be between [0,1]. Typically, not required for MRI, Z-Score
    # Normalization would be more important

    # Histogram: Visual representation of quantitative data distribution
    # Histogram Equalization adjusts the contrast of an image by adjusting histogram to be uniform throughout.

    # Noise Reduction: Removes distortions, grains, speckles while keeping
    # important detail and clarity(Use for testing only)

    # Skull Stripping: Strong potential only if every image model sees is in this format. For deployment,
    # every image must also be preprocessed with Skull Stripping.

    # Intensity Normalization: MRI intensities are not standardized. The same tissue can appear with a different
    # brightness. Intensity Normalization brings the intensities to a common scale in order to separate tissue and
    # segment the brain. For deployment, every image must also be preprocessed with Intensity Normalization.