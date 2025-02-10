# Deep Learning-based Pneumonia Detection System

## Overview
This project implements a deep learning system for automated pneumonia detection using chest X-ray images. It utilizes ResNet-18 architecture and compares two approaches:
1. Training a model from scratch
2. Fine-tuning a pre-trained model

The fine-tuned model achieved 86.2% accuracy on the test set, demonstrating the effectiveness of transfer learning for medical image analysis.

## Dataset
The project uses a subset of the Chest X-Ray Images (Pneumonia) dataset from Kaggle. While the full dataset contains 5,863 grayscale X-ray images, we used a reduced version to accommodate GPU memory constraints and reduce training time:

Original dataset size:
- Training set: 5,216 images
- Validation set: 16 images
- Test set: 624 images

### Dataset Subsampling
To manage computational resources effectively:
- Every 10th image was selected from each set
- This significantly reduced memory usage and training time
- Allowed for faster experimentation and iteration
- Made the project feasible with limited GPU resources

### Data Processing
- Images are resized to 128 × 128 pixels (reduced from original size for memory efficiency)
- Normalization using ImageNet statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225])
- Data augmentation: random horizontal flipping (p=0.5) and rotations (±10°)

## Requirements
