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
```
torch
torchvision
tensorboard
torchcam
PIL
matplotlib
numpy
sklearn
```

## Model Architecture
The system uses ResNet-18 architecture with two implementation approaches:

### Training from Scratch (Task 1.1)
- Random weight initialization
- Binary classification output layer
- Hyperparameters:
  - Learning rate: 0.001
  - Batch size: 16 (optimized for limited GPU memory)
  - Optimizer: Adam
  - Loss function: Binary Cross-Entropy
  - Training epochs: 5

### Transfer Learning (Task 1.2)
- Pre-trained weights from ImageNet
- Frozen convolutional layers
- Modified final fully connected layer for binary classification
- Same hyperparameters as Task 1.1

## Performance

### Model Trained from Scratch
- Overall accuracy: 78.5%
- Normal class accuracy: 76.2%
- Pneumonia class accuracy: 80.8%

### Fine-tuned Model
- Overall accuracy: 86.2%
- Normal class accuracy: 84.5%
- Pneumonia class accuracy: 87.9%

Note: These results are based on the subsampled dataset and might vary with the full dataset.

## Visualization
The project includes:
- Training and validation loss curves tracked via TensorBoard
- Grad-CAM visualizations for model interpretability and failure case analysis

## Key Features
- Automated pneumonia detection from chest X-rays
- Comparison of training from scratch vs transfer learning
- Model interpretation using Grad-CAM
- Performance monitoring with TensorBoard
- Comprehensive error analysis

## Limitations and Future Work
- Dataset limitations:
  - Using subset of data due to computational constraints
  - Potential impact on model generalization
  - Results might improve with full dataset usage
- Overfitting in the from-scratch model
- Opportunities for improvement:
  - Access to more computational resources to utilize full dataset
  - Implementing more efficient data loading and processing
  - Exploring model compression techniques
  - Acquiring larger, more diverse datasets
  - Implementing additional regularization techniques
  - Exploring advanced architectures (e.g., Vision Transformers)
  - Enhancing model interpretability

## References
1. Rudan, Igor, et al. "Epidemiology and etiology of childhood pneumonia." Bulletin of the World Health Organization 86.5 (2008): 408-416B.
2. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
3. Selvaraju, Ramprasaath R., et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.
4. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
5. Kaggle, "Chest X-Ray Images (Pneumonia)," https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Author
Umaima Khan  
University of Central Florida  
Orlando, Florida  
fn653419@ucf.edu
