import glob
import os
import re
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np



# Download latest version
# path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")

# Normal Image Size is 224x224
# Image Size Varies
# 7 images per patient. First image is original while the rest are augmented, so we will only grab the original
# Every 7 images is the original
def collect_image_paths():
    image_paths = []
    labels = []

    def get_images(path, label):
        for file in (glob.iglob(path)):
            filename = os.path.splitext(os.path.basename(file))[0]  # e.g., 'N_1' or 'N_1_BR'
            #if re.fullmatch(r'[A-Z]_\d+', filename):  # e.g., 'N_1', 'G_2', etc
            image_paths.append(file)
            labels.append(label)

    # Load Image Paths
    get_images("./Brain_Tumor_Dataset/Normal/*.jpg", 0)
    get_images("./Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg", 1)
    get_images("./Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg", 1)
    get_images("./Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg", 1)

    # Return Image Paths and Corresponding Labels
    return image_paths, torch.tensor(labels, dtype=torch.float32)


class MRI(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        # Normalize, Resize, and Adjust Tensor Shape
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor()
                                             ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Load Image on Demand
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)

        return image, label

