from train import train
from model import Model
from gradcam import GradCAM
from dataset import MRI, get_data_split, compute
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

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
# image Shape: (224, 224, 1) 224x224, 1 Color Channel

# Activation Function: ReLU
# Optimizer: ADAM

# To Identify Type of Tumor:
# Loss Function: Cross Entropy
# Inputs: 1 Image
# Outputs: 4 - Normal, Glioma, Meningioma, Pituitary

def main():
    torch.manual_seed(51)
    np.random.seed(51)
    random.seed(51)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparameters
    epochs = 39
    batch_size = 16
    learning_rate = 0.00005
    weight_decay = 0.005
    dropout_probability = 0.3
    model = Model(dropout_probability).to(device)

    print(model)
    print(f'Epochs: {epochs}')
    print(f'Batch Size: {batch_size}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Weight Decay: {weight_decay}')
    print(f'Dropout Probability: {dropout_probability}')
    train(epochs, batch_size, learning_rate, weight_decay, model)

    _, _, _, _, X_test, y_test = get_data_split()
    mri = MRI(X_test, y_test)
    cam = GradCAM(model, target_layer_name='cnn.13')

    for i in range(50):
        random_index = random.randint(1, 1238)
        image, label = mri.__getitem__(random_index)
        print(f'Label : {label}')
        image = image.to(device)
        # GradCAM Image
        overlay_image_jet, overlay_image_hot = cam.heatmap_overlay(image, target_class=label)

        prediction = model(image.unsqueeze(0))
        plt.title(f'Label: {label}, Model Prediction: {prediction}')
        # Red Heat Map
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(overlay_image_jet)

        # Blue Heat Map
        plt.subplot(1, 2, 2)
        plt.imshow(overlay_image_hot)
    # Visualizing Feature Maps
    # num_layers = 0
    # conv_layers = []
    # # List of the 2 sequential objects in model (nn.Sequential)
    # model_children = list(model.children())
    #
    # for child in model_children:
    #     # Checks for type Sequential
    #     if type(child) == nn.Sequential:
    #         # Want to visualize the child of model_children
    #         for layer in child.children():
    #             # If it's a Convolutional Layer
    #             if not type(layer) == nn.Linear:
    #                 # Record This Layer
    #                 num_layers += 1
    #                 conv_layers.append(layer)
    #
    # print(conv_layers)
    # print(f'X Shape: {X_train.shape}')
    # dataset = MRI(X_train, y_train, testing=False)
    # image, label = dataset.__getitem__(3233)
    # print(f'Image Shape Before: {image.shape}')
    # print(label)
    # print(image.unsqueeze(0).shape)
    # plt.imshow(image.view(image.shape[2], image.shape[1], image.shape[0]))
    # plt.show()
    # image = image.unsqueeze(0).to(device)
    # results = [conv_layers[0](image)]
    # for i in range(1, len(conv_layers)):
    #     # Input for next layer is output of last layer
    #     results.append(conv_layers[i](results[-1]))
    # print(results[0].shape)
    #
    # # Visualize
    # for layer in range(len(results)):
    #     plt.figure(figsize=(30, 10))
    #     layer_viz = results[layer].squeeze()
    #     for i, f in enumerate(layer_viz):
    #         print(f'F Shape: {f.shape}')
    #         plt.subplot(2, 8, i + 1)
    #         plt.imshow(f.detach().cpu().numpy())
    #         plt.axis('off')
    #     plt.show()

    # Brain MRI Images Visualization

    # # replace=False avoids duplicates values
    # random_index = np.random.choice(len(X), 6, replace=False)  # Choose 6 Random Indexes for MRI Images
    # plt.figure(figsize=(10, 5))
    # # print(f'Random Index: {random_index}')
    # # Show Images
    # for i in range(6):
    #     image, label = dataset[random_index[i]]
    #     image = image.view(image.shape[2], image.shape[1], image.shape[0])
    #     plt.subplot(2, 3, i + 1)
    #     # Matplotlib takes images as (Height, Width, Channel)
    #     plt.imshow(image)  # PyTorch takes images as (Channel, Height, Width)
    #     plt.title(f"Label: {int(y[random_index[i]])}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    #
    # # Histogram Plot
    # plt.figure(figsize=(10, 9))  # Wider and taller figure
    #
    # for i in range(3):
    #     image, label = dataset[random_index[i]]
    #
    #     # Convert from (C, H, W) to (H, W, C) for display
    #     image_np = image.permute(1, 2, 0).numpy()
    #
    #     # Image subplot
    #     plt.subplot(3, 2, 2 * i + 1)
    #     plt.imshow(image_np)
    #     plt.title(f"Image - Label: {int(label)}")
    #     plt.axis('off')
    #
    #     # Histogram subplot
    #     plt.subplot(3, 2, 2 * i + 2)
    #     sns.histplot(image.numpy().ravel(), kde=True, bins=50, color='skyblue')
    #     plt.title("Pixel Intensity Distribution")
    #     plt.xlabel("Pixel Value")
    #     plt.ylabel("Frequency")
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
