# Custom Multi-Label Image Classifier for Produce Detection

## Overview
This project implements a custom deep learning model for multi-label classification of fruits and vegetables in images. It serves as an alternative to YOLO for produce detection, focusing on classification accuracy rather than real-time object detection.

## Purpose
- **Verification**: Validate the accuracy of a custom neural network against traditional object detection models
- **Education**: Demonstrate the implementation of a multi-label classification system from scratch
- **Performance Comparison**: Provide a baseline for comparing with YOLO's performance

## Model Architecture
The model is a deep neural network with the following architecture:
- Input: 64x64x3 RGB images (flattened to 12,288 features)
- Hidden Layers: 
  - Linear(12288, 2048) + ReLU
  - Linear(2048, 1024) + ReLU 
  - Linear(1024, 512) + ReLU
  - Linear(512, 256) + ReLU
  - Linear(256, 128) + ReLU
- Output: Linear(128, 63) with sigmoid activation for multi-label classification

## Performance
- **Accuracy**: ~98.5% on test set
- **Training Time**: Varies by hardware (typically 10-30 minutes on GPU)

## Comparison with YOLO

| Feature | This Model | YOLO v9 |
|---------|------------|---------|
| **Purpose** | Multi-label classification | Object detection + classification |
| **Speed** | Faster inference (classification only) | Slower (detection + classification) |
| **Accuracy** | High for classification | High for detection + classification |
| **Use Case** | When you only need classification | When you need bounding boxes |
| **Training Data** | Requires pre-cropped images | Uses full images with bounding boxes |
| **Output** | Class probabilities | Bounding boxes + class probabilities |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hungh/YoLoL
   cd YoLoL

## When to Use This Model

- **Classification-Only Tasks**: When you only need to classify pre-cropped images
- **Resource-Constrained Environments**: Lower computational requirements than YOLO
- **Educational Purposes**: Simpler architecture for learning and experimentation
- **Baseline Model**: As a reference for comparing with more complex models

## When to Use YOLO

- **Object Detection**: When you need to detect and locate multiple objects
- **Real-Time Applications**: When you need both detection and classification
- **Variable Input Sizes**: When input image sizes vary significantly
- **Production Deployment**: When you need a battle-tested solution

## Trade-offs

### Simplicity vs. Functionality
- **This Model**: Simpler but only handles classification
- **YOLO**: More complex but handles both detection and classification

### Performance
- **This Model**: Faster for pure classification tasks
- **YOLO**: Slower but provides object localization

### Training Data
- **This Model**: Requires pre-processed crops
- **YOLO**: Works with raw images and annotations

### Flexibility
- **This Model**: Fixed input size
- **YOLO**: Handles various input sizes

### Deployment
- **This Model**: Easier to deploy for simple classification
- **YOLO**: Requires more resources but offers more features

## Performance Considerations

### This Model
- Lower memory footprint
- Faster inference for classification
- Easier to train and debug
- Better suited for embedded systems

### YOLO
- Higher accuracy for detection tasks
- More versatile for real-world applications
- Better at handling multiple objects
- More complex to train and optimize

## Conclusion

This custom model serves as an excellent starting point for understanding deep learning for computer vision and provides a solid baseline for classification tasks. However, for production environments requiring object detection, YOLO or similar models would be more appropriate despite their higher computational requirements.