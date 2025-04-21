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

### Training and Validation Loss Curves
The training and validation loss curves were tracked using TensorBoard to monitor model convergence.

**Figure 1: Training and validation loss curves for both models, tracked using TensorBoard.**  
![Training and Validation Loss](https://raw.githubusercontent.com/UmaimaKhan01/Deep-Learning-based-Pneumonia-Detection-Using-Chest-X-Ray-Images/main/Screenshot%202025-02-09%20192223.png)
![Metrics](https://github.com/UmaimaKhan01/Deep-Learning-based-Pneumonia-Detection-Using-Chest-X-Ray-Images/blob/main/Screenshot%202025-02-09%20192111.png?raw=true)

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
