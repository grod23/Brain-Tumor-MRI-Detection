import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



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
            image_paths.append(file)
            labels.append(label)

    # Load Image Paths
    get_images("./Brain_Tumor_Dataset/Normal/*.jpg", 0)
    get_images("./Brain_Tumor_Dataset/Tumor/glioma_tumor/*.jpg", 1)
    get_images("./Brain_Tumor_Dataset/Tumor/meningioma_tumor/*.jpg", 1)
    get_images("./Brain_Tumor_Dataset/Tumor/pituitary_tumor/*.jpg", 1)

    # Return Image Paths and Corresponding Labels
    return image_paths, torch.tensor(labels, dtype=torch.float32)

def z_score_normalization(image):
    # Image Tensor
    mean = image.mean()
    sd = image.std()
    return (image - mean) / sd

class MRI(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        # Min-Max Normalization, Resize, and Adjust Tensor Shape
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor()
                                             ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index, testing=False):
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Load Image on Demand
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Preprocessing

        # Histogram Equalization
        image = cv2.equalizeHist(image)
        # Convert to PIL Image
        image = Image.fromarray(image)
        # Normalize and Resize
        image = self.transform(image)
        # Z-Score Normalization
        image = z_score_normalization(image)

        # For testing:
        # No augmented images
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