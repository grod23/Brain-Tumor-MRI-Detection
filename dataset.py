import glob
import os
import re
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Download latest version
# path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")

class MRI(Dataset):
    def __init__(self):
        # Normal Image Size is 224x224
        # Image Size Varies
        # Load Image Arrays
        normal = self.load_images("./Brain_Tumor_Dataset/Normal/*.jpg")
        glioma = self.load_images("./Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg")
        meningioma = self.load_images("./Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg")
        pituitary = self.load_images("./Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg")

        # Create Image Tensors
        normal = torch.tensor(normal, dtype=torch.float32)
        glioma = torch.tensor(glioma, dtype=torch.float32)
        meningioma = torch.tensor(meningioma, dtype=torch.float32)
        pituitary = torch.tensor(pituitary, dtype=torch.float32)

        # Combine Tumor Tensors
        tumors = torch.cat((glioma, meningioma, pituitary), dim=0)

        # Array Shapes: (Number of Images, 224, 224, 3)
        # (Number of Images, Width, Height, Color Channels)
        # (18606, 224, 224, 3)

        # Tumor: 2658 Images
        # Normal: 438 Images

        # Assign 1 for tumors, 0 for normal
        y_tumor = torch.ones(tumors.shape[0], dtype=torch.float32)
        y_normal = torch.zeros(normal.shape[0], dtype=torch.float32)

        # Combine Data
        X = torch.cat((normal, tumors), dim=0)
        y = torch.cat((y_normal, y_tumor), dim=0)
        # X Image tensor shape: (3096, 224, 224, 3)

        # Normalize
        X = X.permute(0, 3, 1, 2).float() / 255.0  # RGB Values between 0 and 1
        # Shape now (3096, 3, 224, 224) from (3096, 224, 224, 3)

        self.images = X
        self.labels = y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def load_images(self, path):
        images = []
        for file in (glob.iglob(path)):
            filename = os.path.splitext(os.path.basename(file))[0]  # e.g., 'N_1' or 'N_1_BR'
            if re.fullmatch(r'[A-Z]_\d+', filename):  # e.g., 'N_1', 'G_2', etc
                image = cv2.imread(file)
                # Ensure Consistent Image Size
                image = cv2.resize(image, (224, 224))
                # print(image.mode)
                # Ensure Consistent RGB Order
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

        return np.array(images)
