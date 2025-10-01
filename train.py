import sys
import os

# Graphing Library
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Evaluation Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Neural Network Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

# Python Files
from model import Model
from dataset import MRI, get_data_split


def train(epochs, batches, learning_rate, weight_decay, dropout_probability):
    print(f'Device Available: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Model GPU
    model = Model(dropout_probability).to(device)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Couples with Weight Decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # End Training Early if no longer decreasing loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Get Train Val Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = get_data_split()

    # Create DataSet Instances
    train_dataset = MRI(X_train, y_train)
    validation_dataset = MRI(X_val, y_val)
    test_dataset = MRI(X_test, y_test)

    # Create DataLoaders
    training_loader = DataLoader(train_dataset, batch_size=batches, num_workers=4,
                                 shuffle=True)  # Only Shuffle Training Data
    validation_loader = DataLoader(validation_dataset, batch_size=batches, num_workers=4, shuffle=False)
    testing_loader = DataLoader(test_dataset, batch_size=batches, num_workers=4, shuffle=False)
    # Num_workers specify how many parallel subprocesses are used to load the data
    # DataLoaders also add Batch Size to Shape: (batch_size, 1, 224, 224)

    # Loss Tracking
    loss_track = []
    # Validation Tracking
    val_track = []
    # Accuracy
    correct = 0
    total = 0
    # Training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in training_loader:
            # Use GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Reset Gradients
            optimizer.zero_grad()
            # Get y_hat
            y_predicted = model(X_batch)
            # print(f'Prediction: {y_predicted.argmax(dim=1)}')
            # print(f'Label: {y_batch}')
            # Get Loss
            loss = loss_fn(y_predicted, y_batch)  # y_batch must be of type LongTensor()
            # Backpropagation
            loss.backward()
            # Update Learnable Parameters
            optimizer.step()
            # Update Epoch Loss
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(training_loader)
        loss_track.append(train_loss)
        print(f'Training Epoch: {epoch}, Loss: {train_loss}')

        # Validation for the current epoch
        model.eval()
        # No Gradient Calculation
        with torch.no_grad():
            batch_loss = 0
            for X_val, y_val in validation_loader:
                # Use GPU
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_predicted = model(X_val)
                loss = loss_fn(y_predicted, y_val)
                batch_loss += loss.item()
                correct += (y_predicted.argmax(dim=1) == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = batch_loss / len(validation_loader)
        val_track.append(avg_val_loss)
        print(f'Training Epoch: {epoch}, Validation Loss: {avg_val_loss}')

    print(f'Correct: {correct}, Total Images: {total}')
    print(f'Validation Accuracy: {correct / total}')

    # Visualize Training and Validation Loss
    plt.figure(figsize=(15, 9))
    plt.plot(loss_track, c='b', label='Train Loss')
    plt.plot(val_track, c='r', label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.show()

    # Testing

    # Accuracy
    test_correct = 0
    test_total = 0
    # Classification Report
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_test, y_test in testing_loader:
            # Use GPU
            X_test, y_test = X_test.to(device), y_test.to(device)
            # Model Prediction
            y_prediction = model(X_test)

            # Get Total Report
            y_true.extend(y_test.cpu().numpy())
            y_pred.extend(y_prediction.cpu().numpy())

            # Total Accuracy
            test_correct += (y_prediction.argmax(dim=1)==y_test).sum().item()
            test_total += y_test.size(0)


    test_accuracy = test_correct / test_total
    print(f'Correct: {test_correct}, Total: {test_total}')
    print(f'Test Accuracy: {test_accuracy}')

