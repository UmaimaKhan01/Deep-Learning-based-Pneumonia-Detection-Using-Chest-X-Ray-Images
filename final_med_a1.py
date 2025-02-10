# My Chest X-ray Analysis Project
# This program helps doctors classify chest X-rays as either normal or showing signs of pneumonia
# I'm using deep learning (ResNet18) to make this classification automatically

from google.colab import drive
drive.mount('/content/drive')

# Where I'm storing my X-ray images
my_xray_folder = '/content/drive/MyDrive/archive/chest_xray/chest_xray'

# Installing the libraries I need for deep learning and visualization
!pip install torch torchvision tensorboard
!pip install torchcam

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Setting up how I want to process my images before feeding them to the AI
# For training images, I'm adding some random flips and rotations to make the model more robust
image_preprocessing_training = transforms.Compose([
    transforms.Resize((128, 128)),  # Making all images the same size - smaller is faster!
    transforms.RandomHorizontalFlip(),  # Sometimes flip the image to help the model learn better
    transforms.RandomRotation(10),  # Slightly rotate images for more variety
    transforms.ToTensor(),  # Convert to format PyTorch understands
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# For validation and testing, we keep it simple - just resize and normalize
image_preprocessing_evaluation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading my image datasets
training_dataset = datasets.ImageFolder(os.path.join(my_xray_folder, 'train'), transform=image_preprocessing_training)
validation_dataset = datasets.ImageFolder(os.path.join(my_xray_folder, 'val'), transform=image_preprocessing_evaluation)
testing_dataset = datasets.ImageFolder(os.path.join(my_xray_folder, 'test'), transform=image_preprocessing_evaluation)

# I'm using a subset of images to speed up training (every 10th image)
training_subset = Subset(training_dataset, indices=range(0, len(training_dataset), 10))
validation_subset = Subset(validation_dataset, indices=range(0, len(validation_dataset), 10))
testing_subset = Subset(testing_dataset, indices=range(0, len(testing_dataset), 10))

# Setting up my data loaders - they'll feed the images to my model in small batches
batch_size = 16  # How many images to process at once
training_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
testing_loader = DataLoader(testing_subset, batch_size=batch_size, shuffle=False)

# Creating two different models to compare:
# 1. A fresh model that learns everything from scratch
fresh_model = models.resnet18(weights=None)
output_features = fresh_model.fc.in_features
fresh_model.fc = nn.Linear(output_features, 2)  # 2 classes: Normal and Pneumonia

# 2. A pre-trained model that already knows about general images
pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
pretrained_model.fc = nn.Linear(output_features, 2)

# For the pre-trained model, we only want to train the last layer
for param in pretrained_model.parameters():
    param.requires_grad = False
pretrained_model.fc.requires_grad_(True)

# Moving models to GPU if available for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fresh_model.to(device)
pretrained_model.to(device)

def train_my_model(model, loss_function, optimizer, training_epochs=5):
    """
    This is my training function that teaches the model to recognize pneumonia in X-rays
    It keeps track of how well the model is learning and saves the best version
    """
    performance_tracker = SummaryWriter()
    best_validation_accuracy = 0.0

    for epoch in range(training_epochs):
        print(f'Training Day {epoch+1}/{training_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                current_loader = training_loader
            else:
                model.eval()
                current_loader = validation_loader

            total_loss = 0.0
            correct_predictions = 0

            for images, labels in current_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(images)
                    _, predicted_labels = torch.max(predictions, 1)
                    batch_loss = loss_function(predictions, labels)

                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

                total_loss += batch_loss.item() * images.size(0)
                correct_predictions += torch.sum(predicted_labels == labels.data)

            epoch_loss = total_loss / len(current_loader.dataset)
            epoch_accuracy = correct_predictions.double() / len(current_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')
            performance_tracker.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            performance_tracker.add_scalar(f'{phase} Accuracy', epoch_accuracy, epoch)

            if phase == 'val' and epoch_accuracy > best_validation_accuracy:
                best_validation_accuracy = epoch_accuracy
                torch.save(model.state_dict(), 'my_best_model.pth')

    performance_tracker.close()
    return model

# Setting up training parameters
loss_calculator = nn.CrossEntropyLoss()
fresh_model_optimizer = optim.Adam(fresh_model.parameters(), lr=0.001)
pretrained_model_optimizer = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

# Training both models
print("Training my fresh model from scratch:")
trained_fresh_model = train_my_model(fresh_model, loss_calculator, fresh_model_optimizer)

print("Training my pre-trained model:")
trained_pretrained_model = train_my_model(pretrained_model, loss_calculator, pretrained_model_optimizer)

def evaluate_my_model(model, test_loader):
    """
    This function tests how well my model performs on new X-rays it hasn't seen before
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    final_accuracy = correct / total
    print(f'Test Accuracy: {final_accuracy:.4f}')
    print(classification_report(all_true_labels, all_predictions, target_names=['Normal', 'Pneumonia']))
    return final_accuracy

def visualize_mistakes(model, test_loader):
    """
    This function shows us where our model made mistakes and tries to explain why
    It highlights the areas of the X-ray that influenced the model's decision
    """
    important_layer = model.layer4[-1]
    for param in important_layer.parameters():
        param.requires_grad = True

    attention_visualizer = GradCAM(model, target_layer=important_layer)
    wrong_predictions = []

    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        for i in range(len(predictions)):
            if predictions[i] != labels[i]:
                wrong_predictions.append(images[i].cpu())

    # Show the first 5 mistakes
    for image_tensor in wrong_predictions[:5]:
        single_image = image_tensor.unsqueeze(0).to(device)
        output = model(single_image)
        attention_map = attention_visualizer(output.squeeze(0).argmax().item(), output)

        # Converting everything to a format we can display
        image = image_tensor.permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype('uint8')

        if image.shape[-1] == 1:
            image = image.squeeze(-1)
            image_pil = Image.fromarray(image, mode='L')
        else:
            image_pil = Image.fromarray(image)

        attention_map_np = attention_map[0].cpu().numpy()
        if attention_map_np.ndim > 2:
            attention_map_np = attention_map_np.squeeze()

        # Overlay the attention map on the original image
        final_visualization = overlay_mask(
            image_pil,
            Image.fromarray((attention_map_np * 255).astype('uint8'), mode='L'),
            alpha=0.5
        )
        plt.imshow(final_visualization)
        plt.axis('off')
        plt.show()

    for param in important_layer.parameters():
        param.requires_grad = False

# Testing both models
print("Testing my fresh model:")
evaluate_my_model(fresh_model, testing_loader)

print("Testing my pre-trained model:")
evaluate_my_model(pretrained_model, testing_loader)

# Visualizing where my pre-trained model made mistakes
visualize_mistakes(pretrained_model, testing_loader)

# Plotting my training progress
training_losses_fresh = [0.4429, 0.1972, 0.1471, 0.1303, 0.1483]
validation_losses_fresh = [0.4345, 0.1574, 1.5979, 1.9694, 2.7181]

training_losses_pretrained = [0.4325, 0.3137, 0.2550, 0.2434, 0.2340]
validation_losses_pretrained = [0.4967, 0.4211, 0.3392, 0.3222, 0.3518]

plt.figure(figsize=(12, 5))

# Plotting fresh model progress
plt.subplot(1, 2, 1)
plt.plot(training_losses_fresh, label='Training Loss', marker='o')
plt.plot(validation_losses_fresh, label='Validation Loss', marker='x')
plt.title('Learning from Scratch', fontsize=14)
plt.xlabel('Training Day', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Plotting pre-trained model progress
plt.subplot(1, 2, 2)
plt.plot(training_losses_pretrained, label='Training Loss', marker='o')
plt.plot(validation_losses_pretrained, label='Validation Loss', marker='x')
plt.title('Fine-Tuning Pre-trained Model', fontsize=14)
plt.xlabel('Training Day', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
