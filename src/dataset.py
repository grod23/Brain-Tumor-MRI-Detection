# File Handling
import glob
import os.path
import re
import sys

import cv2
import matplotlib.pyplot as plt
import torch
from IPython.core.pylabtools import figsize
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from collections import defaultdict
import imutils


# Download latest version
# path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")

# Normal Image Size is 224x224
# Image Size Varies
# 7 images per patient. First image is original while the rest are augmented
# Every 7 images is the original

# Gets the File Number in order to sort numerically
def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group())

# Loads Images Locally
def collect_image_paths():
    image_paths = []
    labels = []
    groups = []

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
    get_images("C:/Users/gabe7/PycharmProjects/BrainTumorDetectionMRI/Brain_Tumor_Dataset/Normal/*.jpg", 0)
    get_images("C:/Users/gabe7/PycharmProjects/BrainTumorDetectionMRI/Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg", 1)
    get_images("C:/Users/gabe7/PycharmProjects/BrainTumorDetectionMRI/Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg", 2)
    get_images("C:/Users/gabe7/PycharmProjects/BrainTumorDetectionMRI/Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg", 3)

    return np.array(image_paths), np.array(labels), np.array(groups)

# Loads images from s3 bucket.
def collect_s3():
    # SageMaker mounts input data to /opt/ml/input/data/
    base_dir = "/opt/ml/input/data/training"
    image_paths = []
    labels = []
    groups = []

    def get_s3_images(path, label):
        # Image Paths for Specific Patient
        patients = defaultdict(list)
        files = glob.iglob(os.path.join(path, "*.jpg"))
        # Sort Files Numerically
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

    # Use LOCAL paths now
    get_s3_images(os.path.join(base_dir, "Normal"), 0)
    get_s3_images(os.path.join(base_dir, "glioma_tumor"), 1)
    get_s3_images(os.path.join(base_dir, "meningioma_tumor"), 2)
    get_s3_images(os.path.join(base_dir, "pituitary_tumor"), 3)

    print(image_paths, labels, groups)
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

    print(f'X Test: {len(X_test)}')
    print(f'Y test: {len(y_test)}')
    print(f'X Validation: {len(X_val)}')
    print(f'Y Validation: {len(y_val)}')

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


def compute(image_paths, target_size=(224, 224)):
    pixel_values = []

    for path in image_paths:
        # Read grayscale image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Resize to target size (same as model input)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Convert to float32 in [0, 255] (do NOT divide by 255 yet!)
        img = img.astype(np.float32)  # Keep in [0, 255] for stats

        pixel_values.append(img.flatten())

    # Now all flattened arrays have same length: 224*224 = 50176
    all_pixels = np.concatenate(pixel_values)

    mean = np.mean(all_pixels) / 255.0
    std = np.std(all_pixels) / 255.0

    return mean, std

# Will crop and resize image
# Image already grayscale
def crop_image(image, path):
    # Reduce noise and smooth image.
    new_image = cv2.GaussianBlur(image,(5, 5), 0)

    # Converts image to binary (black and white) pixels above 45 are white(255). Pixels below 45 become black(0).
    # Isolates bright regions in MRI(brain tissue).
    new_image = cv2.threshold(new_image, 28, 255, cv2.THRESH_BINARY)[1]

    # Morphological Operations to smooth and clean binary mask:
    # Helps see contours easily, removes small white noise.
    new_image = cv2.erode(new_image, None, iterations=2)
    # Expands white areas back to original size.
    new_image = cv2.dilate(new_image, None, iterations=2)

    # Send as copy so data is not lost, retrieves only outermost contours(RETR_EXTERNAL).
    contours = cv2.findContours(new_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    # Safety Check
    if not contours:
        print("[WARNING] No contours found!")
        print(path)
        return image

    # Get the largest contour by measuring the area. The largest contour should be the brain tissue.
    contours = max(contours, key=cv2.contourArea)
    # Bounding boxes of contour/brain tissue. Need left, right, top, and bottom.
    ext_left = tuple(contours[contours[:, : , 0].argmin()])[0]
    ext_right = tuple(contours[contours[:, :, 0].argmax()])[0]
    ext_top = tuple(contours[contours[:, :, 1].argmin()])[0]
    ext_bottom = tuple(contours[contours[:, :, 1].argmax()])[0]

    # Slice the image through rectangular bounding box.
    cropped_image = image[ext_top[1]: ext_bottom[1], ext_left[0]: ext_right[0]]
    return cropped_image

class MRI(Dataset):
    def __init__(self, image_paths, labels, testing=False):
        self.image_paths = image_paths
        self.labels = labels
        # Resize, Adjust Tensor Shape, Min-Max Normalization, Z-Score Normalization
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224, 224)),
                                             # Small shifts
                                             # transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), fill=0),
                                             transforms.ToTensor()
                                             # transforms.Normalize(mean=[0.19], std=[0.19])
                                             ])
        self.testing = testing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        label = self.labels[index]

        # Load Image on Demand
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Crop Image
        image = crop_image(image, image_path)

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