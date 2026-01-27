# Custom Multi-Label Image Classifier for Produce Detection

## Overview
This project implements a custom deep learning model for multi-label classification of fruits and vegetables in images. It serves as an alternative to YOLO for produce detection, focusing on classification accuracy rather than real-time object detection.

## Purpose
- **Learning & Development**: Build deep learning expertise through hands-on implementation
- **CNN Implementation**: Develop convolutional neural networks for image classification
- **Skill Development**: Generate mental trace and deep understanding of CNN architectures
- **Educational Journey**: Progress from classic NN to advanced CNN techniques
- **Performance Exploration**: Train on full YOLO dataset to understand CNN capabilities

## Model Architecture (classic implementation for CPU training)
### Phase 1: Classic Neural Network (Completed)
- **Input**: Flattened image features
- **Hidden Layers**: Dense layers with ReLU activation
- **Output**: Multi-label classification with sigmoid
### Phase 2: Convolutional Neural Network (Current)
- **Input**: Full YOLO dataset images (maintaining spatial structure)
- **Architecture**: CNN layers with convolution, pooling, and dense layers
- **Goal**: Learn CNN implementation and training on real-world image data
- **Focus**: Skill development rather than beating YOLO performance
### Phase 3: Directly use the YoLo library
- **Input**: Full YOLO dataset images (maintaining spatial structure)
- **Architecture**: Use pre-trained YoLo model for classification
- **Goal**: Learn how to use pre-trained models and fine-tune them for specific tasks
- **Focus**: Skill development rather than beating YOLO performance
## CNN Implementation Goals
- **Educational**: Master CNN concepts through practical implementation
- **Hands-on Experience**: Work with real image datasets and CNN architectures
- **Deep Learning Skills**: Develop comprehensive understanding of:
  - Convolutional layers and feature extraction
  - Pooling layers and spatial dimensionality reduction
  - CNN-specific optimization techniques
  - Image preprocessing and augmentation
- **Mental Trace**: Build strong foundation for future deep learning projects
## Training Dataset
- **Source**: Full YOLO image dataset
- **Scale**: Large-scale image classification task
- **Complexity**: Real-world produce images with varying conditions
- **Purpose**: Practice CNN implementation on substantial dataset
## Performance Expectations
- **Primary Goal**: Learning and skill development
- **Secondary**: Achieve reasonable classification performance
- **Benchmark**: Personal improvement and understanding rather than YOLO competitio

## Model Architecture (using PyTorch with CUDA support)
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
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

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
