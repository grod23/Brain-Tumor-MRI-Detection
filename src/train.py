import sys
import os
# Graphing Library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Evaluation Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Neural Network Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

# Python Files
from model import Model
from dataset import MRI, get_data_split
import argparse

def train(epochs, batch_size, learning_rate, weight_decay, dropout_probability):
    print(f'Device Available: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Model GPU
    model = Model(dropout_probability).to(device)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Couples with Weight Decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # End Training Early if no longer decreasing loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Get Train Val Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = get_data_split()

    # Create DataSet Instances
    train_dataset = MRI(X_train, y_train)
    validation_dataset = MRI(X_val, y_val)
    test_dataset = MRI(X_test, y_test)

    # Create DataLoaders
    training_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                 shuffle=True)  # Only Shuffle Training Data
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    testing_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    # Num_workers specify how many parallel subprocesses are used to load the data
    # DataLoaders also add Batch Size to Shape: (batch_size, 1, 224, 224)

    # Loss Tracking
    loss_track = []
    # Validation Tracking
    val_track = []
    # Training
    for epoch in range(epochs):
        # Accuracy
        correct = 0
        total = 0
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
        print(f'Epoch: {epoch}, Training Loss: {train_loss}')

        # Validation for the current epoch
        model.eval()
        # No Gradient Calculation
        with torch.no_grad():
            epoch_loss = 0
            for X_val, y_val in validation_loader:
                # Use GPU
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_predicted = model(X_val)
                loss = loss_fn(y_predicted, y_val)
                # Track Loss
                epoch_loss += loss.item()
                # Track Accuracy
                correct += (y_predicted.argmax(dim=1) == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = epoch_loss / len(validation_loader)
        # Update Learning Rate
        scheduler.step(avg_val_loss)
        # Track Validation Loss
        val_track.append(avg_val_loss)
        # Print Statements
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
            y_pred.extend(y_prediction.argmax(dim=1).cpu().numpy())

            # Total Accuracy
            test_correct += (y_prediction.argmax(dim=1)==y_test).sum().item()
            test_total += y_test.size(0)


    test_accuracy = test_correct / test_total
    print(f'Correct: {test_correct}, Total: {test_total}')
    print(f'Test Accuracy: {test_accuracy}')

    # Evaluation Metrics
    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    # Get unique labels to use for axis labels
    labels = sorted(list(set(y_true + y_pred)))
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

    # Display Confusion Matrix Heat Map
    plt.figure(figsize=(8,6))
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    # Classification Report
    print(report)

if __name__ == '__main__':
    # Sagemaker Compatible
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout_probability', type=float, default=0.3)

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_probability=args.dropout_probability
    )